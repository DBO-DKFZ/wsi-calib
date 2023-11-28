from typing import Union, Optional
import os
from pathlib import Path
import yaml
import warnings

import torch

from ..datamodules import histo_module


class Run:
    def __init__(
        self,
        exp_dir: str,
        checkpoint_name="final",
        run_dir: str = os.environ["EXPERIMENT_LOCATION"],
        pred_path: str = "predictions",
        config_path: str = "logs/config.yaml",
        name: Union[str, None] = None,
        lazy: bool = True,
        properties: Optional[dict] = None,
    ):
        exp_path = Path(run_dir) / exp_dir
        config_path = exp_path / config_path

        pred_file_names = {
            "val": f"test_val_{checkpoint_name}.preds",
            "test_id": f"test_id_{checkpoint_name}.preds",
            "test_ood": f"test_ood_{checkpoint_name}.preds",
        }

        preds_path = {key: exp_path / pred_path / value for key, value in pred_file_names.items()}

        self.exp_path = exp_path
        self.pred_path = exp_path / pred_path
        assert exp_path.exists()
        assert config_path.exists()

        if properties is not None:
            self.properties = properties
            for key, value in properties.items():
                setattr(self, key, value)

        # for _, path in preds_path.items():
        #    assert path.exists(), f"Path {str(path)} does not exists"

        self._config = None
        self._preds_path = preds_path
        self._preds = {key: None for key in preds_path}

        self.config_file = config_path

        self.name = str(exp_dir) + "/" + checkpoint_name if name is None else name

        if not lazy:
            self.val_preds
            self.test_id_preds
            self.test_ood_preds

    @property
    def val_preds(self):
        if self._preds["val"] is None:
            self._preds["val"] = torch.load(str(self._preds_path["val"]))

        return self._preds["val"]

    @property
    def test_id_preds(self):
        if self._preds["test_id"] is None:
            self._preds["test_id"] = torch.load(str(self._preds_path["test_id"]))

        return self._preds["test_id"]

    @property
    def test_ood_preds(self):
        if self._preds["test_ood"] is None:
            self._preds["test_ood"] = torch.load(str(self._preds_path["test_ood"]))

        return self._preds["test_ood"]

    @property
    def config(self):
        if self._config is None:
            self._config = yaml.safe_load(open(str(self.config_file), "r"))

        return self._config

    def get_preds(self, split: str):
        if split == "test_id":
            return self.test_id_preds
        elif split == "test_ood":
            return self.test_ood_preds
        else:
            path = self.pred_path / (split + ".preds")
            if path.exists():
                preds = torch.load(str(path))
                return preds
            else:
                RuntimeError("Split must be in [test_id, test_ood] or {str(path)'} must exist!")

    def get_preds_and_dataset(self, split: str):
        assert split in ["test_id", "test_ood"]

        data_config = self.config["data"].copy()
        class_name = data_config["class_path"].split(".")[-1]
        class_args = data_config["init_args"]

        # get correct class
        if class_name == "HistoSlideModule":
            data_class = getattr(globals()["histo_module"], class_name)
        else:
            raise RuntimeError("Could not find corresponding class definition for ", class_name)

        # Replace paths if trained on cluster or workstation.
        general_data_path = Path(os.environ["DATASET_LOCATION"])
        if "MCO-SCalib" in class_args["data_p"]:
            class_args["data_p"] = str(general_data_path / "MCO-SCalib")
        elif "TCGA-SCalib" in class_args["data_p"]:
            class_args["data_p"] = str(general_data_path / "TCGA-SCalib")
        if "TCGA-CRC-SCalib" in class_args["data_ood_p"]:
            class_args["data_ood_p"] = str(general_data_path / "TCGA-CRC-SCalib")

        data_module = data_class(**class_args)

        # Get the correct dataset and predictions
        if split in ["test_id", "test_ood"]:
            data_module.setup("test")
            test_dataloaders = data_module.test_dataloader()
            if type(test_dataloaders) in [list, tuple]:
                test_id, test_ood = test_dataloaders
            else:
                test_id = test_dataloaders
                test_ood = None

            if split == "test_ood" and test_ood is not None:
                dataset = test_ood.dataset
                preds = self.test_ood_preds
            else:
                dataset = test_id.dataset
                preds = self.test_id_preds

        return preds, data_module, dataset


def create_runs_from_folder(
    exp_dir: Path,
    checkpoint: str = "final",
    name: Optional[str] = None,
    max_num: Optional[int] = None,
    run_dir=os.environ["EXPERIMENT_LOCATION"],
    config_path: str = "logs/config.yaml",
    prediction_path="predictions",
    properties=None,
) -> list[Run]:
    runs = []

    dir = Path(run_dir) / Path(exp_dir)

    if not dir.exists():
        raise RuntimeError(f"Provided path {dir} does not exist.")

    configs = sorted(list(dir.glob(f"**/{config_path}")))

    if max_num is not None:
        configs = configs[-max_num:]

    if len(configs) == 0:
        warnings.warn(f"Found no runs in {dir}.")

    for config in configs:
        config = Path(config)

        assert config.parts[-2] == "logs"

        tmp_dir = config.parents[1].relative_to(run_dir)

        runs.append(
            Run(
                tmp_dir,
                checkpoint_name=checkpoint,
                pred_path=prediction_path,
                config_path=config_path,
                run_dir=run_dir,
                name=name,
                properties=properties,
            )
        )

    return runs


def extract_results(results):
    if isinstance(results, list):
        if len(results) == 2:
            out_sm, y = results
        elif len(results) == 3:
            out_sm, y, var_sm = results
        else:
            print("Unknown length of results")
            return -1
        return out_sm, y

    if isinstance(results, dict):
        return results["softmax"], results["label"]
