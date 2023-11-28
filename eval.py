from typing import Optional, Union
import os
import sys
from pathlib import Path
import itertools
import argparse
import jsonargparse

from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from torch import inference_mode

from src.utils.custom_cli import CustomCLI
from src.utils.custom_writer import CustomWriter
from src.callbacks.temperature_scaling import TemperatureScalingCallback


def run_eval(cli: CustomCLI, version_dir: Path, model_checkpoint: Union[str, Path]):
    # Make directories to write logs and predictions
    eval_dir = version_dir / "eval_logs"
    if cli.config["output"] is not None:
        output_dir = version_dir / cli.config["output"]
    else:
        output_dir = version_dir / "predictions"
    output_dir.mkdir(exist_ok=True, parents=True)

    # Remove lighning_logs subdirectory
    # tensorboard_logger = TensorBoardLogger(save_dir=str(eval_dir), name=None, version="")
    csv_logger = CSVLogger(save_dir=str(eval_dir), name=None, version="")

    writer_callback = CustomWriter(output_dir=output_dir, write_interval="epoch", name=Path(model_checkpoint).stem)

    loggers = [csv_logger]
    callbacks = [writer_callback]

    # TODO: Optionally adjust datamodule args

    cli.instantiate_classes(trainer=False)
    datamodule = cli.datamodule

    datamodule.setup("test")
    val_dataloader = datamodule.val_dataloader()

    test_dataloaders = datamodule.test_dataloader()
    if type(test_dataloaders) in [list, tuple]:
        test_id_dataloader, test_ood_dataloader = test_dataloaders
    else:
        test_id_dataloader = test_dataloaders
        test_ood_dataloader = None

    if cli.config["temp_scaling"] == True:
        ts_callback = TemperatureScalingCallback(validation_loader=val_dataloader, function_name="forward_logits")
        callbacks.append(ts_callback)
        # Temperature Scaling needs to have gradients enabled in trainer.test(), this is why we need to set inference_mode=False
        cli.instantiate_trainer(add_callbacks=callbacks, logger=loggers, inference_mode=False)
    else:
        # Normal Trainer instantiation with default inference_mode=True
        cli.instantiate_trainer(add_callbacks=callbacks, logger=loggers)

    # Assign trainer from CLI
    trainer = cli.trainer

    # ckpt_save = config.get("checkpoint")
    model = cli.model.load_from_checkpoint(
        checkpoint_path=model_checkpoint,
        **cli.config["model"]["init_args"],  # Overwrite attributes from config
    )

    print(f"Writing predictions to {output_dir}")

    # Add metric suffix for writing predictions
    model.metric_suffix = "id"
    trainer.test(model, dataloaders=test_id_dataloader)
    if test_ood_dataloader is not None:
        model.metric_suffix = "ood"
        trainer.test(model, dataloaders=test_ood_dataloader)


def main_eval():
    def add_arguments_to_parser(parser: jsonargparse.ArgumentParser):
        parser.add_argument("--checkpoint", required=True)

        # TODO: Avoid copying arguments
        # Copy arguments from train.py
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
        parser_kwargs={"parser_mode": "omegaconf", "error_handler": None},
        save_config_kwargs={"overwrite": True},  # Needed to perform multiple test runs
        # auto_class=True,  # Currently not implemented
    )

    config = cli.config
    config_path = Path(str(cli.config["config"][0]))
    model_checkpoint = Path(config.checkpoint)

    assert config_path.parts[-2] == "logs"
    version_dir = config_path.parents[1]

    if not model_checkpoint.exists():
        raise RuntimeError("Provided .ckpt does not exist")

    run_eval(cli, version_dir, model_checkpoint)


def main(**kwargs):
    p = argparse.ArgumentParser()
    p.add_argument("--eval_dir", type=str, nargs="+", default=False)
    p.add_argument("--version", type=str, nargs="+", default=False)
    p.add_argument("--checkpoint", default="final")

    a, rest = p.parse_known_args()

    if a.eval_dir is not False:
        path_list = a.eval_dir
        ckpt = a.checkpoint
        assert ckpt in ["last", "final", "last.ckpt", "final.ckpt"]
        if not ".ckpt" in ckpt:
            ckpt += ".ckpt"

        if not (isinstance(path_list, list) or isinstance(path_list, tuple)):
            path_list = [path_list]

        # Turn into Paths objects
        for i in range(len(path_list)):
            path_list[i] = Path(path_list[i])
            assert path_list[i].exists()

        # Build Cartesian Product
        if a.version is not False:
            versions = a.version
            if not (isinstance(versions, list) or isinstance(versions, tuple)):
                versions = [versions]

            tmp_path_list = []
            for exp_dir, version in itertools.product(path_list, versions):
                if version == "null":
                    tmp_path_list.append(exp_dir)
                else:
                    tmp_path_list.append(exp_dir / f"version_{version}")
                # print(tmp_path_list[-1])
                assert tmp_path_list[-1].exists()

            path_list = tmp_path_list

        found_ckpts = []

        for exp_path in path_list:
            exp_path = Path(exp_path)
            assert exp_path.exists(), f"Path {exp_path} does not exist."

            tmp_ckpts = [Path(i) for i in exp_path.glob(f"**/{ckpt}")]
            found_ckpts.extend(tmp_ckpts)

        # Search for a config for every checkpoint
        found_confs = []
        for ckpt in found_ckpts:
            assert ckpt.parents[0].stem == "checkpoints"

            config_file = ckpt.parents[1] / "logs" / "config.yaml"
            assert config_file.exists()

            found_confs.append(config_file)

        save_sysargv = sys.argv.copy()

        for ckpt, conf in zip(found_ckpts, found_confs):
            print(f"Evaluating {conf}")
            sys.argv = (
                [sys.argv[0]] + ["--config", str(conf), "--checkpoint", str(ckpt)] + rest
            )  # [sys.argv[0]] + ["--config", str(conf), "--checkpoint", str(ckpt)] + sys.argv[1:]

            main_eval()

            sys.argv = save_sysargv
    else:
        main_eval()


if __name__ == "__main__":
    main()
