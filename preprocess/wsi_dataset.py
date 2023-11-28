# STL
from typing import Optional, Tuple
import os
from pathlib import Path

# CPU
import pandas as pd
import numpy as np
import PIL
from PIL import Image

# Backends
import openslide
import cucim

# GPU
import torch
from torch.utils import data
from torchvision import transforms


class OpenSlideReader:
    def __init__(self, slide_p: Path, process_level: int = 0):
        self.slide = openslide.OpenSlide(slide_p)
        self.process_level = process_level

    def read_region(self, location: Tuple[int, int], size: Tuple[int, int]) -> Image.Image:
        img = self.slide.read_region(
            location=location,
            level=self.process_level,
            size=size,
        )
        img = img.convert("RGB")  # Convert from RGBA to RGB
        return img


class CuCIMReader:
    def __init__(self, slide_p: Path, process_level: int = 0):
        self.slide = cucim.CuImage(str(slide_p))
        self.process_level = process_level

    def read_region(self, location: Tuple[int, int], size: Tuple[int, int]) -> np.ndarray:
        img = self.slide.read_region(
            location=location,
            size=size,
            level=self.process_level,
        )
        img = np.asarray(img)  # Convert CuImage to np.ndarray
        return img


class WSIPatches(data.Dataset):
    default_transform = transforms.Compose(
        [
            transforms.ToTensor(),  # Important to first transform to tensor to deal with different data types!
        ]
    )

    def __init__(
        self,
        slide_p: Path,
        patch_p: Path,
        process_level: int = 0,
        transform: transforms.Compose = default_transform,
        backend: str = "openslide",
    ):
        self.patch_p = patch_p
        self.slide_name = patch_p.stem
        info_p = patch_p / "tile_information.csv"
        assert info_p.exists()
        df = pd.read_csv(info_p)
        self.df = df
        self.transform = transform

        if backend == "openslide":
            self.reader = OpenSlideReader(
                slide_p=slide_p,
                process_level=process_level,
            )
        elif backend == "cucim":
            self.reader = CuCIMReader(
                slide_p=slide_p,
                process_level=process_level,
            )
        else:
            raise RuntimeError("Provided unknown backend")

    def __len__(self):
        return len(self.df)

    def _read_region_at_loc(self, idx):
        entry = self.df.loc[idx]
        img = self.reader.read_region(
            location=(entry["x_pos"], entry["y_pos"]),
            size=(entry["patch_size_x"], entry["patch_size_y"]),
        )
        return img, (entry["x_pos"], entry["y_pos"])

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        img, coord = self._read_region_at_loc(idx)
        img = self.transform(img)  # transform image to tensor with shape (C, H, W)
        coord = torch.tensor(coord)
        return img, coord

    def get_img(self, idx) -> Image.Image:
        img, _ = self._read_region_at_loc(idx)
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        return img


def load_slide_openslide(slide_p: Path) -> Image.Image:
    slide = openslide.OpenSlide(slide_p)
    level_count = slide.level_count
    level_dimensions = slide.level_dimensions
    min_size = level_dimensions[level_count - 1]  # Index from 0 - level_count-1
    img = slide.get_thumbnail(size=min_size)
    return img


def load_slide_cucim(slide_p: Path) -> Image.Image:
    slide = cucim.CuImage(str(slide_p))
    resolutions = slide.resolutions
    level_count = resolutions["level_count"]
    level_idx = level_count - 1
    size = resolutions["level_dimensions"][level_idx]
    cu_img = slide.read_region(
        location=(0, 0),
        size=size,
        level=level_idx,
    )
    tensor = torch.tensor(np.asarray(cu_img))
    img = Image.fromarray(np.asarray(cu_img))
    return img


if __name__ == "__main__":
    slide_p = (
        Path(os.environ["SLIDE_ROOT_DIR"])
        / "TCGA-COAD"
        / "TCGA-3L-AA1B-01Z-00-DX1.8923A151-A690-40B7-9E5A-FCBEDFC2394F.svs"
    )
    thumbnail = load_slide_cucim(slide_p)
