from pathlib import Path

from omegaconf import DictConfig

from .data import AMPDataModule
from .tasks import DiffusionTraining


def instantiate_datamodule(config: DictConfig, split_seed: int):
    if config.data.name == "amp-embeddings":
        return AMPDataModule(Path(config.data.original_amp_file),
                             Path(config.data.embeddings_file),
                             batch_size=config.task.batch_size,
                             split_seed=split_seed,
                             computable_conditioning_names=config.data.computable_conditioning_names,
                             uncomputable_conditioning_names=config.data.uncomputable_conditioning_names,
                             only_amps=config.data.only_amps)

def instantiate_task(config: DictConfig, datamodule):
    if config.task.name == "diffusion-training":
        return DiffusionTraining(datamodule=datamodule,
                                 dim=config.model.hidden_dim, 
                                 beta_schedule=config.model.beta_schedule,
                                 timesteps=config.model.timesteps,
                                 max_epochs=config.trainer.max_epochs,
                                 dim_mults=config.model.dim_mults,
                                 self_condition=config.model.self_condition,
                                 objective=config.model.objective,
                                 rescaled_phi=config.model.rescaled_phi,
                                 tau1=config.model.tau1,
                                 tau2=config.model.tau2,
                                 noise_strength=config.model.noise_strength,
                                 variable_conditioning=config.model.variable_conditioning,
                                 loss_weighting=config.model.loss_weighting,
                                 learning_rate=config.task.learning_rate,
                                 min_learning_rate=config.task.min_learning_rate,
                                 lr_decay=config.task.lr_decay,
                                 optimizer=config.task.optimizer,
                                 no_test_samples=config.task.no_test_samples,
                                 evaluation_model_embedding_dim=config.task.evaluation_model_embedding_dim,
                                 classifier_model_path=config.task.classifier_model_path,
                                 encoder_decoder_model=config.task.encoder_decoder,
                                 aa_scales=config.task.aa_scales,
                                 sample_batch_size=config.task.sample_batch_size,
                                 no_saved_samples=config.no_saved_samples,
                                 sample_save_path=config.sample_save_path,
                                 )


def load_conditioning(config: DictConfig):
    datamodule = instantiate_datamodule(config, config.seed)
    datamodule.prepare_data()
    datamodule.setup("train")
    return datamodule.dataset.conditioning

def load_model_for_inference(config, checkpoint_path):
    datamodule = instantiate_datamodule(config, config.seed)
    datamodule.prepare_data()
    datamodule.setup("train")
    
    # Load model from checkpoint
    model = DiffusionTraining.load_from_checkpoint(checkpoint_path, 
                                                    datamodule=datamodule,
                                                    dim=config.model.hidden_dim, 
                                                    beta_schedule=config.model.beta_schedule,
                                                    timesteps=config.model.timesteps,
                                                    max_epochs=config.trainer.max_epochs,
                                                    dim_mults=config.model.dim_mults,
                                                    self_condition=config.model.self_condition,
                                                    objective=config.model.objective,
                                                    rescaled_phi=config.model.rescaled_phi,
                                                    tau1=config.model.tau1,
                                                    tau2=config.model.tau2,
                                                    noise_strength=config.model.noise_strength,
                                                    variable_conditioning=config.model.variable_conditioning,
                                                    learning_rate=config.task.learning_rate,
                                                    min_learning_rate=config.task.min_learning_rate,
                                                    lr_decay=config.task.lr_decay,
                                                    optimizer=config.task.optimizer,
                                                    no_test_samples=config.task.no_test_samples,
                                                    evaluation_model_embedding_dim=config.task.evaluation_model_embedding_dim,
                                                    classifier_model_path=config.task.classifier_model_path,
                                                    aa_scales=config.task.aa_scales,
                                                    encoder_decoder_model=config.task.encoder_decoder,
                                                    sample_batch_size=config.task.sample_batch_size,
                                                    no_saved_samples=config.no_saved_samples,
                                                    sample_save_path=config.sample_save_path,
                                                    )
    model.eval()
    return model