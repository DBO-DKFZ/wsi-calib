from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from sklearn.model_selection import KFold
from torchsampler import ImbalancedDatasetSampler

from .datasets.histo import HistoSlides


class HistoSlideModule(LightningDataModule):
    def __init__(
        self,
        data_p: str,
        feature_folder: str,
        patch_folder: str,
        task: str,
        batch_size: int,
        num_workers: int,
        fold: Optional[int] = 1,  # fold number from 1 - n_folds
        data_ood_p: Optional[str] = None,
    ):
        super().__init__()
        self.data_p = Path(data_p)
        self.feature_folder = feature_folder
        self.patch_folder = patch_folder
        self.data_ood_p = Path(data_ood_p) if data_ood_p is not None else None
        if task == "msi":
            self.label_dict = {"mss": 0, "msi": 1}
        else:
            raise NotImplementedError

        # Infer n_folds from number of files in folds directory
        self.fold_p = self.data_p / "folds"
        assert self.fold_p.exists()
        fold_files = list(self.fold_p.iterdir())
        n_folds = int(len(fold_files) / 2)  # Two files (train/val) for each fold

        if fold is not None:
            assert 1 <= fold <= n_folds, "Invalid fold number"
        self.fold = fold

        self.save_hyperparameters(logger=False)

    def setup(self, stage: str = None):
        dataset_args = {
            "root_p": self.data_p,
            "feature_folder": self.feature_folder,
            "patch_folder": self.patch_folder,
            "label_dict": self.label_dict,
        }

        if stage in ["fit", "validate", "test", "predict"]:
            # Need validation dataset in test step for Temperature Scaling

            print("Generating train and val datasets...")
            if self.fold is not None:
                # Load fold indices from file
                train_df = pd.read_csv(self.fold_p / f"fold_{self.fold}_train.csv")
                val_df = pd.read_csv(self.fold_p / f"fold_{self.fold}_val.csv")

                self.data_train = HistoSlides(**dataset_args, split="train_val").generate_subset(
                    split="train", indices=train_df["index"].to_list()
                )
                self.data_val = HistoSlides(**dataset_args, split="train_val").generate_subset(
                    split="val", indices=val_df["index"].to_list()
                )
                # Make sure samples in train and val do not overlap
                assert (
                    len(
                        set(self.data_train.df["slide_id"].to_list()).intersection(
                            self.data_val.df["slide_id"].to_list()
                        )
                    )
                    == 0
                )
            else:
                self.data_train = HistoSlides(**dataset_args, split="train_val")
                self.data_val = None

        if stage in ["test", "predict"]:
            print("Generating test datasets...")
            self.data_test_id = HistoSlides(**dataset_args, split="test")
            if self.data_ood_p is not None:
                self.data_test_ood = HistoSlides(
                    root_p=self.data_ood_p,
                    feature_folder=self.feature_folder,
                    patch_folder=self.patch_folder,
                    label_dict=self.label_dict,
                    split="all",
                )

    def train_dataloader(self):
        return DataLoader(
            self.data_train,
            batch_size=self.hparams.batch_size,
            # shuffle=True,  # Mutually exclusive with sampler
            sampler=ImbalancedDatasetSampler(self.data_train),
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.data_val,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            drop_last=False,
        )

    def test_dataloader(self):
        test_id = DataLoader(
            self.data_test_id,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            drop_last=False,
        )

        if self.data_ood_p is not None:
            test_ood = DataLoader(
                self.data_test_ood,
                batch_size=self.hparams.batch_size,
                shuffle=False,
                num_workers=self.hparams.num_workers,
                pin_memory=True,
                drop_last=False,
            )
            return test_id, test_ood
        else:
            return test_id


# Avoid executing scripts within a package.
# See https://stackoverflow.com/questions/16981921/relative-imports-in-python-3 for reference

if __name__ == "__main__":
    from datasets.histo import HistoSlides

    datamodule = HistoSlideModule(
        root_p="/mnt/ssd/Data/MCO-CLAM",
        task="msi",
        batch_size=16,
        num_workers=8,
        fold=1,
        seed=42,
    )
    datamodule.setup("fit")
    print("Train Dataset:")
    print(datamodule.train_dataloader().dataset.df.head(n=10))
    print(len(datamodule.train_dataloader().dataset))
    print("Val Dataset:")
    print(datamodule.val_dataloader().dataset.df.head())
    print(len(datamodule.val_dataloader().dataset))
    print("Test Dataset:")
    print(datamodule.test_dataloader().dataset.df.head())
    print(len(datamodule.test_dataloader().dataset))
