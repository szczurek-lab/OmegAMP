#!/usr/bin/env python

import faulthandler
import logging
import warnings

import hydra
import wandb
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    TQDMProgressBar,
)
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.trainer.connectors.signal_connector import _SignalConnector


from project.config import instantiate_datamodule, instantiate_task
from project.utils import (
    WandbModelCheckpoint,
    WandbSummaries,
    filter_device_available,
    get_logger,
    log_hyperparameters,
    print_config,
    print_exceptions,
    set_seed,
)
from project.utils.time_limit import TimeLimit

# Log to traceback to stderr on segfault
faulthandler.enable(all_threads=False)

# If data loading is really not a bottleneck for you, uncomment this to silence the
# warning about it
# warnings.filterwarnings(
#     "ignore",
#     "The '\w+_dataloader' does not have many workers",
#     module="lightning",
# )
logging.getLogger("pytorch_lightning.utilities.rank_zero").addFilter(
    filter_device_available
)


log = get_logger()

class Null_SignalConnector(_SignalConnector):
    def register_signal_handlers(self):
        pass

def get_callbacks(config):
    monitor = {"monitor": config.monitor, "mode": "max"}
    callbacks = [
        WandbSummaries(**monitor),
        WandbModelCheckpoint(
            save_last=True, save_top_k=1, every_n_epochs=1, filename="best", **monitor
        ),
        TQDMProgressBar(refresh_rate=1),
        LearningRateMonitor(logging_interval="step")
    ]
    if config.early_stopping is not None:
        stopper = EarlyStopping(
            patience=int(config.early_stopping),
            min_delta=0,
            strict=False,
            check_on_train_epoch_end=False,
            **monitor,
        )
        callbacks.append(stopper)
    if config.get("train_limit") is not None:
        callbacks.append(TimeLimit(config.train_limit))
    return callbacks


@hydra.main(config_path="config", config_name="train", version_base=None)
@print_exceptions
def main(config: DictConfig):
    
    # Resolve interpolations to work around a bug:
    # https://github.com/omry/omegaconf/issues/862
    OmegaConf.resolve(config)
    wandb.init(**config.wandb, resume=(config.wandb.mode == "online") and "allow")
    print_config(config)

    log.info("Loading data")
    datamodule = instantiate_datamodule(config, config.seed)
    datamodule.prepare_data()
    datamodule.setup("train")

    log.info("Instantiating model and task")
    task = instantiate_task(config, datamodule)

    logger = WandbLogger()
    log_hyperparameters(logger, config, task)

    log.info("Instantiating trainer")
    callbacks = get_callbacks(config)
    trainer: Trainer = instantiate(config.trainer, callbacks=callbacks, logger=logger)

    # submitit handles the requeuing, so we disable pytorch-lightning's SLURM feature
    trainer.signal_connector = Null_SignalConnector(trainer)

    log.info("Starting training!")
    trainer.fit(task, datamodule=datamodule)

    if config.eval_testset:
        log.info("Starting testing!")
        trainer.test(ckpt_path="best", datamodule=datamodule)

    wandb.finish()
    log.info(f"Best checkpoint path:\n{trainer.checkpoint_callback.best_model_path}")

    best_score = trainer.checkpoint_callback.best_model_score
    return float(best_score) if best_score is not None else None


if __name__ == "__main__":
    main()
