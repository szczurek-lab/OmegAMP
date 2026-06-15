from collections import namedtuple
import torch.nn as nn
import torch
from functools import partial
import numpy as np
import math
from tqdm.auto import tqdm
from torch.cuda.amp import autocast
from einops import rearrange, reduce
import scipy.optimize as so
import torch.nn.functional as F
from random import random

# constants

ModelPrediction =  namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

# helpers functions

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def cast_tuple(t, length = 1):
    if isinstance(t, tuple):
        return t
    return ((t,) * length)

def divisible_by(numer, denom):
    return (numer % denom) == 0

def identity(t, *args, **kwargs):
    return t

# normalization functions

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5


# gaussian diffusion trainer class

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def linear_beta_schedule(timesteps):
    """
    linear schedule, proposed in original ddpm paper
    """
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)

def log_linear_beta_schedule(timesteps):
    """A version of the linear beta schedule that works for arbitrary timesteps."""

    log_alphas_cumprod_T = np.log(1e-6)
    T, log_T = timesteps, np.log(timesteps)
    one_to_T = np.arange(1, T + 1)

    def f(alpha_T):
        return (
            np.log(T + one_to_T * (alpha_T - 1)).sum() - T * log_T - log_alphas_cumprod_T
        )

    alpha_T = so.bisect(f, 1e-10, 1.0)
    alphas = (T + one_to_T * (alpha_T - 1)) / T
    betas = 1 - alphas
    return torch.tensor(betas)


def log_snr_linear_beta_schedule(timesteps, snr_1=1e3, snr_T=1e-5):
    """A beta schedule that decays the log-SNR linearly."""

    T = timesteps
    log_snr_1 = np.log(snr_1)
    log_snr_T = np.log(snr_T)

    alpha_cumprods = []
    for t in range(1, T + 1):

        def f(alpha_cumprod):
            return (
                np.log(alpha_cumprod)
                - np.log1p(-alpha_cumprod)
                - ((T - t) * log_snr_1 + (t - 1) * log_snr_T) / (T - 1)
            )

        alpha_cumprods.append(so.bisect(f, 1e-8, 1.0 - 1e-8))
    alpha_cumprods = np.array(alpha_cumprods)

    alphas = np.concatenate(
        (alpha_cumprods[:1], alpha_cumprods[1:] / alpha_cumprods[:-1])
    )
    betas = 1 - alphas
    return torch.tensor(betas)


def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

def sigmoid_beta_schedule(timesteps, start = -3, end = 3, tau = 1, clamp_min = 1e-5):
    """
    sigmoid schedule
    proposed in https://arxiv.org/abs/2212.11972 - Figure 8
    better for images > 64x64, when used during training
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

def get_annealing_schedule(tau1, tau2, timesteps):
    tau1_scaled = tau1 * (timesteps - 1)
    tau2_scaled = tau2 * (timesteps - 1)

    timestep_tensor = torch.arange(timesteps)

    gamma_values = torch.zeros(timesteps)

    mask1 = timestep_tensor <= tau1_scaled
    gamma_values[mask1] = 1.0

    mask2 = (timestep_tensor > tau1_scaled) & (timestep_tensor < tau2_scaled)
    gamma_values[mask2] = (tau2_scaled - timestep_tensor[mask2]) / (tau2_scaled - tau1_scaled)

    return gamma_values

def adjust_with_reconstruction_guidance(x_denoised, x_target, mask, guidance_strength, clamp = 0.1):
    x_denoised = x_denoised.clone().detach().requires_grad_(True)

    batch_target = x_target.expand_as(x_denoised)
    batch_mask = mask.expand_as(x_denoised)
    batch_target = batch_target.to(x_denoised.device)
    batch_mask = batch_mask.to(x_denoised.device)
    batch_size = x_denoised.shape[0]

    reconstruction_loss = F.mse_loss(x_denoised * batch_mask, batch_target * batch_mask, reduction = 'sum')
    
    gradient, = torch.autograd.grad(reconstruction_loss, x_denoised)

    updated_x_denoised = x_denoised - guidance_strength * batch_mask * torch.clamp(gradient, min=-clamp, max=clamp)

    return updated_x_denoised


class GaussianDiffusion1D(nn.Module):
    def __init__(
        self,
        model,
        *,
        seq_length,
        timesteps = 1000,
        sampling_timesteps = None,
        objective = 'pred_noise',
        beta_schedule = 'cosine',
        ddim_sampling_eta = 0.,
        auto_normalize = False,
        rescaled_phi = 0.7,
        noise_strength = 0.1,
        tau1 = 0.01,
        tau2 = 1
    ):
        super().__init__()
        self.model = model
        self.channels = self.model.channels
        self.self_condition = self.model.self_condition

        self.seq_length = seq_length

        self.objective = objective
        
        annealing_schedule = get_annealing_schedule(tau1, tau2, timesteps)

        assert objective in {'pred_noise', 'pred_x0', 'pred_v'}, 'objective must be either pred_noise (predict noise) or pred_x0 (predict image start) or pred_v (predict v [v-parameterization as defined in appendix D of progressive distillation paper, used in imagen-video successfully])'

        if beta_schedule == "linear":
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == "log-linear":
            betas = log_linear_beta_schedule(timesteps)
        elif beta_schedule == "log-snr-linear":
            betas = log_snr_linear_beta_schedule(timesteps)
        elif beta_schedule == "cosine":
            betas = cosine_beta_schedule(timesteps)
        elif beta_schedule == "sigmoid":
            betas = sigmoid_beta_schedule(timesteps)
        else:
            raise ValueError(f"unknown beta schedule {beta_schedule}")

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        # sampling related parameters

        self.sampling_timesteps = default(sampling_timesteps, timesteps) # default num sampling timesteps to number of timesteps at training

        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        # helper function to register buffer from float64 to float32

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        register_buffer('annealing_schedule', annealing_schedule)
        register_buffer('rescaled_phi', torch.tensor(rescaled_phi, dtype = torch.float32))
        register_buffer('noise_strength', torch.tensor(noise_strength, dtype = torch.float32))

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # calculate loss weight

        snr = alphas_cumprod / (1 - alphas_cumprod)

        if objective == 'pred_noise':
            loss_weight = torch.ones_like(snr)
        elif objective == 'pred_x0':
            loss_weight = snr
        elif objective == 'pred_v':
            loss_weight = snr / (snr + 1)

        register_buffer('loss_weight', loss_weight)

        # whether to autonormalize

        self.normalize = normalize_to_neg_one_to_one if auto_normalize else identity
        self.unnormalize = unnormalize_to_zero_to_one if auto_normalize else identity

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def predict_v(self, x_start, t, noise):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def predict_start_from_v(self, x_t, t, v):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(self, x, t, classes, x_self_cond = None, clip_x_start = False, rederive_pred_noise = False):
        model_output = self.model.forward_with_cads(x, t, classes, 
                                                    self.annealing_schedule,
                                                    rescaled_phi=self.rescaled_phi, 
                                                    x_self_cond=x_self_cond,
                                                    noise_strength=self.noise_strength)
        maybe_clip = partial(torch.clamp, min = -1., max = 1.) if clip_x_start else identity

        if self.objective == 'pred_noise':
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            x_start = maybe_clip(x_start)

            if clip_x_start and rederive_pred_noise:
                pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_x0':
            x_start = model_output
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_v':
            v = model_output
            x_start = self.predict_start_from_v(x, t, v)
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        return ModelPrediction(pred_noise, x_start)

    def p_mean_variance(self, x, t, classes, x_self_cond = None, clip_denoised = True):
        preds = self.model_predictions(x, t, classes, x_self_cond=x_self_cond)
        x_start = preds.pred_x_start

        if clip_denoised:
            x_start.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start = x_start, x_t = x, t = t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.no_grad()
    def p_sample(self, x, t: int, classes, x_self_cond = None, clip_denoised = True):
        b, *_, device = *x.shape, x.device
        batched_times = torch.full((b,), t, device = x.device, dtype = torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(x = x, t = batched_times, classes = classes, x_self_cond=x_self_cond, clip_denoised = clip_denoised)
        noise = torch.randn_like(x) if t > 0 else 0. # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    @torch.no_grad()
    def p_sample_loop(self, classes, shape, reference_img=None, reference_mask=None, guidance_strength=None):
        batch, device = shape[0], self.betas.device

        img = torch.randn(shape, device=device)

        x_start = None

        for t in tqdm(reversed(range(0, self.num_timesteps)), desc = 'sampling loop time step', total = self.num_timesteps):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(img, t, classes, self_cond)
            if reference_img is not None and reference_mask is not None and guidance_strength is not None:
                with torch.enable_grad():
                    img = adjust_with_reconstruction_guidance(img, reference_img, reference_mask, guidance_strength=guidance_strength)

        img = self.unnormalize(img)
        return img

    @torch.no_grad()
    def p_sample_loop_from_image_and_timestep(self, initial_img, initial_timestep, classes, shape, reference_img=None, reference_mask=None, guidance_strength=None):
        batch, device = shape[0], self.betas.device

        t = torch.full((batch,), initial_timestep, device=device)

        img = self.q_sample(initial_img, t)

        x_start = None

        for t in tqdm(reversed(range(0, initial_timestep)), desc = 'sampling loop time step', total = initial_timestep):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(img, t, classes, self_cond)
            if reference_img is not None and reference_mask is not None and guidance_strength is not None:
                with torch.enable_grad():
                    img = adjust_with_reconstruction_guidance(img, reference_img, reference_mask, guidance_strength=guidance_strength)

        img = self.unnormalize(img)
        return img

    @torch.no_grad()
    def sample(self, conditioning, initial_img=None, initial_timestep=None, batch_size = 16, reference_img = None, reference_mask = None, guidance_strength = None):
        seq_length, channels = self.seq_length, self.channels
        if initial_img is None:
            return self.p_sample_loop(conditioning, (batch_size, channels, seq_length), reference_img=reference_img, reference_mask=reference_mask, guidance_strength=guidance_strength)
        else:
            return self.p_sample_loop_from_image_and_timestep(initial_img, initial_timestep, conditioning, (batch_size, channels, seq_length), 
                                                              reference_img=reference_img, reference_mask=reference_mask, guidance_strength=guidance_strength)

    @torch.no_grad()
    def interpolate(self, x1, x2, t = None, lam = 0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.full((b,), t, device = device)
        xt1, xt2 = map(lambda x: self.q_sample(x, t = t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2

        x_start = None

        for i in tqdm(reversed(range(0, t)), desc = 'interpolation sample time step', total = t):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(img, i, self_cond)

        return img

    @autocast(enabled = False)
    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_losses(self, x_start, t, conditioning, noise = None, class_loss_weights = None):
        b, c, n = x_start.shape
        noise = default(noise, lambda: torch.randn_like(x_start))

        # noise sample

        x = self.q_sample(x_start = x_start, t = t, noise = noise)

        # if doing self-conditioning, 50% of the time, predict x_start from current set of times
        # and condition with unet with that
        # this technique will slow down training by 25%, but seems to lower FID significantly

        x_self_cond = None
        if self.self_condition and random() < 0.5:
            with torch.no_grad():
                x_self_cond = self.model_predictions(x, t, conditioning).pred_x_start
                x_self_cond.detach_()

        # predict and take gradient step

        model_out = self.model.forward_with_cads(x, t, conditioning, self.annealing_schedule,
                                                 rescaled_phi=self.rescaled_phi, x_self_cond=x_self_cond, 
                                                 noise_strength=self.noise_strength)

        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        elif self.objective == 'pred_v':
            v = self.predict_v(x_start, t, noise)
            target = v
        else:
            raise ValueError(f'unknown objective {self.objective}')

        loss = F.mse_loss(model_out, target, reduction = 'none')
        
        loss = reduce(loss, 'b ... -> b', 'mean')

        if class_loss_weights is not None:
            loss = loss * class_loss_weights

        loss = loss * extract(self.loss_weight, t, loss.shape)

        return loss.mean()

    def forward(self, img, *args, **kwargs):
        b, c, n, device, seq_length, = *img.shape, img.device, self.seq_length
        assert n == seq_length, f'seq length must be {seq_length}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        img = self.normalize(img)
        return self.p_losses(img, t, *args, **kwargs)
    