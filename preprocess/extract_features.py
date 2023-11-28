# STL
from typing import Optional, Tuple, Sequence, Union
import sys
import os
from pathlib import Path
from tqdm import tqdm
import json

# GPU Libraries
import torch
import torch.utils.data as data
import torch.nn as nn
import torchvision
from torchvision import transforms

# Custom
from wsi_dataset import WSIPatches


VERIFIED_MODELS = [
    "resnet18",
    "resnet18-ciga",
    "resnet34",
    "resnet50",
]


@torch.no_grad()
def extract_features(
    dataloader: data.DataLoader,
    pretrained_model: nn.Module,
    device: torch.device,
    refresh_rate: int = 1,
) -> Sequence[torch.Tensor]:
    # Move model to GPU
    pretrained_model = pretrained_model.to(device)

    # Set model to eval mode
    # With context manager @torch.no_grad(), no gradients are computed, see
    # https://stackoverflow.com/questions/51748138/pytorch-how-to-set-requires-grad-false
    # Details on how Pytorch Lighning implements inference can be found here:
    # https://lightning.ai/docs/pytorch/stable/common/lightning_module.html
    pretrained_model.eval()
    # for p in pretrained_model.parameters():
    #     p.requires_grad = False

    # Create a tqdm progress bar
    total_batches = len(dataloader)
    pbar = tqdm(total=total_batches, desc="Inference Progress", unit="batches")
    # Set up empty lists
    extracted_features = []
    coordinates = []
    for batch_idx, batch in enumerate(dataloader):
        imgs, coords = batch
        imgs = imgs.to(device)
        feats = pretrained_model(imgs)
        extracted_features.append(feats)
        coordinates.append(coords)
        if refresh_rate != 0 and batch_idx % refresh_rate == 0:
            pbar.update(refresh_rate)
    pbar.update(total_batches)
    extracted_features = torch.cat(extracted_features, dim=0)
    extracted_features = extracted_features.detach().cpu()
    coordinates = torch.cat(coordinates, dim=0)
    coordinates = coordinates.detach().cpu()

    return extracted_features, coordinates


def load_model_weights(model, weights):
    model_dict = model.state_dict()
    weights = {k: v for k, v in weights.items() if k in model_dict}
    if weights == {}:
        print("No weight could be loaded..")
    model_dict.update(weights)
    model.load_state_dict(model_dict)

    return model


def load_ciga_model(checkpoint_p: Path):
    model = torchvision.models.__dict__["resnet18"](weights=None)

    state = torch.load(checkpoint_p, map_location="cuda:0")

    state_dict = state["state_dict"]
    for key in list(state_dict.keys()):
        state_dict[key.replace("model.", "").replace("resnet.", "")] = state_dict.pop(key)

    model = load_model_weights(model, state_dict)

    return model


def main(
    # If type conversion happens within function, all types must be noted at type hinting
    slide_root_dir: Union[str, Path],
    patch_dir: Union[str, Path],
    feat_dir: Union[str, Path],
    dataset_name: str,
    extensions: list[str],
    resize_val: int,
    crop_val: int,
    model: str,
    batch_size: int,
    num_workers: int,
    backend: str,
    pbar_refresh_rate: int,
    *args,
    **kwargs,
):
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    print("Device:", device)

    slide_root_dir = Path(slide_root_dir)
    assert slide_root_dir.exists()
    patch_dir = Path(patch_dir)
    assert patch_dir.exists()
    feat_dir = Path(feat_dir)
    assert feat_dir.exists()
    print("Writing features to %s" % str(feat_dir))

    if dataset_name == "TCGA-CRC":
        subdirs = [
            "TCGA-COAD",
            "TCGA-READ",
        ]
    elif dataset_name == "MCO":
        subdirs = [
            "MCO0001-1000",
            "MCO1001-2000",
            "MCO2001-3000",
            "MCO3001-4000",
            "MCO4001-5000",
            "MCO5001-6000",
            "MCO6001-7000",
        ]
    else:
        raise RuntimeError("Unknown dataset name")

    slide_list = []
    for subdir in subdirs:
        path = slide_root_dir / subdir
        assert path.is_dir()
        for extension in extensions:
            slide_list.extend(path.glob("*" + extension))

    predict_transform = transforms.Compose(
        [
            transforms.ToTensor(),  # Important to have ToTensor transform first to work with PIL.Image and Numpy Array
            transforms.Resize((resize_val, resize_val)),
            transforms.CenterCrop((crop_val, crop_val)),
            # Use ImageNet statistics for normalizing
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    if model == "resnet18-ciga":
        script_p = Path(os.path.abspath(__file__))
        base_p = script_p.parent.parent
        pretrained_model = load_ciga_model(base_p / "checkpoints" / "tenpercent_resnet18.ckpt")
        pretrained_model.fc = torch.nn.Sequential()  # Disable classsification layer
    elif model in torchvision.models.__dict__.keys():
        pretrained_model = torchvision.models.__dict__[model](weights="DEFAULT")
        pretrained_model.fc = torch.nn.Sequential()  # Disable classsification layer
    else:
        raise RuntimeError("Model %s is not available" % model)

    patchp_list = sorted(list(patch_dir.iterdir()))
    # Filter out all elements that are not directories
    patchp_list = [path for path in patchp_list if path.is_dir()]
    featp_list = sorted(list(feat_dir.iterdir()))
    featf_list = [path.stem for path in featp_list]

    # Iterate over all directories where patch locations are stored
    for idx in range(len(patchp_list)):
        patch_p = patchp_list[idx]
        slide_name = patch_p.stem
        with open(patch_p / "slide_info.json") as f:
            info_dict = json.load(f)
        slide_filename = info_dict["slide_filename"]
        paths = [path for path in slide_list if path.name == slide_filename]
        if len(paths) == 1:
            slide_p = paths[0]
        else:
            print("No unique file found for slide %s" % slide_name)
            continue

        # Skip slides with previously extracted features
        if slide_name in featf_list:
            print("Features already extracted. Skipping slide %s (%d/%d)" % (slide_name, idx + 1, len(patchp_list)))
            continue

        # Extract features
        print("Extracting features for slide %s (%d/%d)" % (slide_name, idx + 1, len(patchp_list)))
        try:
            # Include dataset creation in try to catch empty tile_information.csv files
            patch_dataset = WSIPatches(
                slide_p,
                patch_p,
                transform=predict_transform,
                backend=backend,
            )
            patch_dataloader = data.DataLoader(
                patch_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=False,
                drop_last=False,
            )
            feats, coords = extract_features(patch_dataloader, pretrained_model, device, pbar_refresh_rate)
        except KeyboardInterrupt:  # Handle KeyboardInterrupt
            sys.exit(0)
        except Exception:  # Log all other exceptions
            print("An exception occured. Continuing with next slide.")
            # Write log of broken slides to file
            out_p = feat_dir.parent
            with open(out_p / "broken_slides.txt", "a") as f:
                f.write(slide_name + "\n")
            continue  # Do not write output dict to file

        print("Features shape: ", feats.shape)
        output = {"features": feats, "coords": coords}
        torch.save(output, feat_dir / (slide_name + ".pt"))


if __name__ == "__main__":
    from jsonargparse import ArgumentParser, ActionConfigFile

    parser = ArgumentParser(parser_mode="omegaconf", description="Feature Extraction")
    parser.add_argument("--config", action=ActionConfigFile)
    parser.add_argument("--dataset_name", type=str, required=True, choices=["TCGA-CRC", "MCO"])
    parser.add_argument("--slide_root_dir", type=str, required=True)
    parser.add_argument("--patch_dir", type=str, required=True)
    parser.add_argument("--feat_base_dir", type=str, required=True)
    parser.add_argument("--patch_size", type=int, required=True)
    parser.add_argument("--extensions", type=list[str], default=[".svs"])
    parser.add_argument("--resize_val", type=int, default=256)
    parser.add_argument("--crop_val", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=int(os.cpu_count() / 2))
    parser.add_argument("--model", type=str, choices=VERIFIED_MODELS, default="resnet18")
    parser.add_argument("--backend", type=str, choices=["openslide", "cucim"], default="cucim")
    parser.add_argument("--pbar_refresh_rate", type=int, default=10)
    args = parser.parse_args()
    cfg = vars(args)

    # Adjust config
    cfg.pop("config")
    feat_dir = Path(cfg["feat_base_dir"]) / ("features_" + str(cfg["patch_size"]) + "_" + str(cfg["model"]))
    feat_dir.mkdir(exist_ok=True)  # Make directory if directory does not yet exist
    parser.save(cfg=cfg, path=str(feat_dir / "config.yaml"), overwrite=True)
    cfg.update({"feat_dir": str(feat_dir)})

    main(**cfg)
