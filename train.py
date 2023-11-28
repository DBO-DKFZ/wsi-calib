import os
from pathlib import Path
from typing import Optional
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import jsonargparse

from src.utils.custom_cli import CustomCLI
from src.utils.system import find_max_version

from eval import run_eval

METRIC_MODE = {
    "val_loss": "min",
    "val_acc": "max",
    "val_b_acc": "max",
}


def count_labels(dataloader: DataLoader):
    labels = []
    for batch in tqdm(dataloader):
        data, label = batch
        labels.append(label)
    labels = torch.cat(labels)
    unique, counts = torch.unique(labels, return_counts=True)
    print(dict(zip(unique, counts)))


def main(**kwargs):
    def add_arguments_to_parser(parser: jsonargparse.ArgumentParser):
        parser.add_argument("--name", required=True)
        parser.add_argument("--version", type=Optional[int], default=None)
        parser.add_argument("--save_dir", default=os.environ["EXPERIMENT_LOCATION"])
        parser.add_argument("--checkpoint_metric", default="val_b_acc")
        parser.add_argument("--eval", type=bool, default=False)
        parser.add_argument("--output", type=str, default=None)
        parser.add_argument("--temp_scaling", type=bool, default=False)

    cli = CustomCLI(
        run=False,
        instantiate=False,
        add_arguments_function=add_arguments_to_parser,
        parser_kwargs={"parser_mode": "omegaconf", "error_handler": None, "env_prefix": None},
        save_config_kwargs={"overwrite": True},  # Needed to perform multiple test runs
        # auto_class=True,  # Currently not implemented
    )

    cli.instantiate_classes(trainer=False)
    config = cli.config

    # Test datamodule
    cli.datamodule.setup("fit")
    train_data = cli.datamodule.train_dataloader().dataset
    print("Train samples: %d" % len(train_data))

    # print("Counting labels:")
    # count_labels(cli.datamodule.train_dataloader())

    # Custom setup of trainer
    add_callbacks = []
    add_trainer_args = {}

    # TODO: Create experiment name from config hparams

    experiment_base_path = Path(config.save_dir) / config.name
    if config["version"] is None:
        version = find_max_version(experiment_base_path) + 1
    else:
        version = config["version"]

    version_dir = experiment_base_path / f"version_{version}"
    checkpoint_dir = version_dir / "checkpoints"

    tensorboard_logger = TensorBoardLogger(str(experiment_base_path), name=None, version=version, sub_dir="logs")
    csv_logger = CSVLogger(str(experiment_base_path), name=None, version=version)

    print("Writing logs to " + str(version_dir))
    loggers = [tensorboard_logger]

    assert config.checkpoint_metric in METRIC_MODE.keys()

    model_checkpoint = ModelCheckpoint(
        dirpath=version_dir / "checkpoints",
        filename="{epoch}-{" + str(config.checkpoint_metric) + ":.4f}",
        monitor=config.checkpoint_metric,
        mode=METRIC_MODE[config.checkpoint_metric],
        save_top_k=1,
        save_last=True,
        save_on_train_epoch_end=True,
        save_weights_only=True,
    )

    # Dont save if using optuna
    if not "optuna_trial" in kwargs:
        add_callbacks.extend([model_checkpoint])
    else:
        add_trainer_args["enable_checkpointing"] = False

    if "optuna_trial" in kwargs:
        import optuna

        optuna_callback = optuna.integration.PyTorchLightningPruningCallback(
            kwargs["optuna_trial"], monitor=kwargs["optuna_target"]
        )

        add_callbacks.extend([optuna_callback])

    # Instantiate Trainer
    cli.instantiate_trainer(
        add_callbacks=add_callbacks,
        logger=loggers,
        # num_sanity_val_steps=0, # Moved to config
        # reload_dataloaders_every_n_epochs=1, # Moved to config
        # weights_save_path=checkpoint_dir,
        **add_trainer_args,
    )
    trainer = cli.trainer
    # print("Trainer callbacks:")
    # [print(callback.__class__.__name__) for callback in trainer.callbacks]
    trainer.fit(model=cli.model, datamodule=cli.datamodule)

    # Support for optuna
    if "optuna_trial" in kwargs:
        return trainer.callback_metrics[kwargs["optuna_target"]].item()

    best_model = model_checkpoint.best_model_path

    # Finalize by copying best save to final.ckpt. This marks the experiment as completed.
    if best_model == "":
        # Case for training without validation set
        last_ckpt = checkpoint_dir / "last.ckpt"
        if last_ckpt.exists():
            os.link(last_ckpt, checkpoint_dir / "final.ckpt")
            best_model = checkpoint_dir / "final.ckpt"
    elif best_model is not None and trainer.is_global_zero:
        os.link(best_model, checkpoint_dir / "final.ckpt")
        best_model = checkpoint_dir / "final.ckpt"
    else:
        raise RuntimeError("Cannot handle best_model_path")

    if cli.config["eval"] or cli.config["eval"] == "eval" and trainer.is_global_zero:
        run_eval(cli, version_dir, best_model)


if __name__ == "__main__":
    main()
