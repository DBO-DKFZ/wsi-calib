from typing import Optional, Tuple
from pathlib import Path
import json
import math

import torch
from torch.utils.data import Dataset

import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_coords(coords: torch.Tensor, slidename: str, colors: Optional[torch.Tensor] = None):
    coords = coords.numpy()
    if isinstance(colors, torch.Tensor):
        colors = colors.numpy()
    x, y = coords[:, 0], coords[:, 1]
    y_max = max(y)
    fig, ax = plt.subplots(1, 1)
    pos = ax.scatter(
        x=x,
        y=(y_max - y),  # y-Coordinate is counted down from upper left corner, need to adjust for scatter plot
        c=colors,
    )
    ax.axis("equal")
    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes("right", size="5%", pad=0.05)
    if colors is not None:
        fig.colorbar(pos, ax=ax)
    y_lim = int(math.ceil(y_max / 1000) * 1000)
    # ax.set_ylim(0, y_lim)  # Apparently contradicting with ax.axis("equal")
    ax.set_title(slidename)


class HistoSlides(Dataset):
    def __init__(
        self,
        root_p: Path,
        feature_folder: str,
        patch_folder: str,
        label_dict: dict,
        split: str = "all",
    ):
        super().__init__()
        self.root_p = root_p
        csv_p = self.root_p / "slide_information.csv"
        assert csv_p.exists()
        feat_p = self.root_p / feature_folder
        assert feat_p.exists()
        patch_p = self.root_p / patch_folder
        assert patch_p.exists()
        self.patch_p = patch_p  # Needed for slide visualization

        df_in = pd.read_csv(csv_p)
        assert "Pseudonym" and "Label" in df_in.columns
        file_list = sorted(list(feat_p.glob("*.pt")))  # Only include .pt files
        self.label_dict = label_dict

        # Check sample file
        slidedata = torch.load(file_list[0])
        assert "features" and "coords" in slidedata.keys()

        # Create dataframe
        self.df = self._create_df(df_in, file_list)

        # Check split
        assert split in ["train_val", "test", "all"], "Need to provide a valid split argument"
        self.split = split

        # Adjust dataframe
        if split in ["train_val", "test"]:
            self._adjust_df(split)

    def _create_df(self, df_in: pd.DataFrame, file_list: list) -> pd.DataFrame:
        items = []
        for path in file_list:
            slide_id = path.stem
            assert slide_id in df_in["Pseudonym"].to_list(), (
                "No information about file " + str(path) + " in slide_information.csv"
            )
            df_entry = df_in[df_in["Pseudonym"] == slide_id]
            label = self.label_dict[df_entry["Label"].item()]
            items.append(
                {
                    "feat_path": str(path),
                    "slide_id": slide_id,
                    "label": label,
                }
            )
        return pd.DataFrame(items)

    def _adjust_df(self, split: str):  # Private function with single leading underscore according to PEP8
        test_p = self.root_p / "test_slides.csv"
        assert test_p.exists()
        test_df = pd.read_csv(test_p)
        assert "slide_id" in test_df.columns
        test_ids = test_df["slide_id"].to_list()
        if split == "test":
            self.df = self.df[self.df["slide_id"].isin(test_ids)].reset_index(drop=False)
        elif split == "train_val":
            self.df = self.df[~self.df["slide_id"].isin(test_ids)].reset_index(drop=False)  # Use tilde to negate isin()
        else:
            raise RuntimeError("Provided inavlid split argument")

    def generate_subset(self, split: str, indices: list[int]) -> Dataset:
        self.split = split
        # Extract relevant indices from original index list that contains all samples
        self.df = self.df[self.df["index"].isin(indices)].reset_index(drop=True)
        return self

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        path = Path(self.df["feat_path"][idx])
        label = self.df["label"][idx]
        slidedata = torch.load(path)
        features = slidedata["features"]
        coords = slidedata["coords"]
        return features, coords, label

    def get_dict(self, idx) -> dict:
        path = Path(self.df["feat_path"][idx])
        slidename = self.df["slide_id"][idx]
        label = self.df["label"][idx]
        slidedata = torch.load(path)
        features = slidedata["features"]
        coords = slidedata["coords"]
        item_dict = {
            "slidename": slidename,
            "label": label,
            "features": features,
            "coords": coords,
        }
        return item_dict

    def get_thumbnail(self, idx) -> Tuple[Image.Image, dict]:
        slidename = self.df["slide_id"][idx]
        patch_folder = self.patch_p / slidename
        img = Image.open(patch_folder / "thumbnail.jpeg")
        with open(patch_folder / "slide_info.json") as f:
            info_dict = json.load(f)
        return img, info_dict

    def get_label(self, idx) -> int:
        return self.df["label"][idx]

    def get_labels(self) -> list[int]:  # Required for ImbalancedDatasetSampler
        return self.df["label"].to_list()


if __name__ == "__main__":
    data_p = Path("/mnt/ssd/Data/TCGA-CRC-SCalib")
    feature_folder = "features_resnet18-ciga"
    label_dict = {"mss": 0, "msi": 1}
    dataset = HistoSlides(data_p, feature_folder, label_dict, split="all")
    print(len(dataset))
    item_dict = dataset.get_dict(0)
    print(item_dict["coords"].shape)
    feature_norm = item_dict["features"].norm(dim=1)
    plot_coords(item_dict["coords"], item_dict["slidename"], colors=feature_norm)
    plt.show()
