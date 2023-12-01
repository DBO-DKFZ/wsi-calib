import os
from pathlib import Path
from typing import Union, Tuple

from torchmetrics.functional import roc
from torchmetrics import (
    Accuracy,
    F1Score,
    ConfusionMatrix,
    PrecisionRecallCurve,
    Precision,
    Recall,
    AUROC,
    AveragePrecision,
    CalibrationError,
)
from sklearn.metrics import brier_score_loss
import torch
import numpy as np
import pandas as pd
from PIL import Image, ImageOps

import matplotlib
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns

from src.metrics.adaptive_calibration_error import AdaptiveCalibrationError
from src.eval.visualizations_base import Run, create_runs_from_folder, extract_results
from src.utils.system import RC_DICT, insert_rc_args


def plot_roc_curves(runs: list[Run]) -> Axes:
    dfs = []
    for i in range(len(runs)):
        preds = runs[i].test_id_preds
        Y_prob, labels = preds["softmax"], preds["label"]
        fpr, tpr, thresholds = roc(Y_prob[:, 1], labels, task="binary")
        df = pd.DataFrame(
            {
                "idx": i,
                "fpr": fpr.numpy(),
                "tpr": tpr.numpy(),
                "thresholds": thresholds.numpy(),
            }
        )
        dfs.append(df)
    df = pd.concat(dfs)
    axs = sns.lineplot(data=df, x="fpr", y="tpr", hue="idx")
    return axs


def compute_metrics(
    runs: list[Run],
    split: Union[str, list[str]],
    metrics=None,
    compute_significance=False,
    plot_x="split",
    plot_y="value",
    plot_hue="method",
    plot_col="metric",
    rc_args: dict = None,
    color_palette: str = "colorblind",
    custom_legend: bool = False,
    figname: str = None,
    tabname: str = None,
) -> sns.FacetGrid:
    if isinstance(split, str):
        splits = [split]
    else:
        splits = split

    results = []

    for split in splits:
        # print(f"Computing {split} metrics.")

        for run in runs:
            if split == "val":
                run_preds = run.val_preds
            elif split == "test_id":
                run_preds = run.test_id_preds
            elif split == "test_ood":
                run_preds = run.test_ood_preds
            else:
                raise RuntimeError(f"Undefined split {split}")
            out_sm, label = extract_results(run_preds)

            # Remap split for nice reading
            print_split = {
                "val": "validation",
                "test_id": "ID data",
                "test_ood": "OOD data",
            }[split]

            metric_dict = {}
            metric_dict["method"] = run.name
            metric_dict["split"] = print_split
            metric_dict["run"] = run.exp_path

            if hasattr(run, "properties"):
                for key, value in run.properties.items():
                    metric_dict[key] = value

            if "Accuracy" in metrics:
                acc = Accuracy(task="multiclass", num_classes=2, average="micro")(out_sm, label)
                metric_dict["Accuracy"] = acc.item()

            if "Balanced Accuracy" in metrics:
                b_acc = Accuracy(task="multiclass", num_classes=2, average="macro")(out_sm, label)
                metric_dict["Balanced Accuracy"] = b_acc.item()

            if "F1" in metrics:
                f1 = F1Score(task="multiclass", num_classes=2, ignore_index=0)(out_sm, label)
                metric_dict["F1"] = f1.item()

            if "AUROC" in metrics:
                auroc = AUROC(task="multiclass", num_classes=2)(out_sm, label)
                metric_dict["AUROC"] = auroc.item()

            if "AUCPR" in metrics:
                aucpr = AveragePrecision(task="multiclass", num_classes=2)(out_sm, label)
                metric_dict["AUCPR"] = aucpr.item()

            if "Precision" in metrics:
                pre = Precision(task="multiclass", num_classes=2, ignore_index=0)(out_sm, label)
                metric_dict["Precision"] = pre.item()

            if "Recall" in metrics:
                rec = Recall(task="multiclass", num_classes=2, ignore_index=0)(out_sm, label)
                metric_dict["Recall"] = rec.item()

            if "ECE" in metrics:
                ece = CalibrationError(task="binary", n_bins=15, norm="l1")(out_sm[:, 1], label)
                metric_dict["ECE"] = ece.item()

            if "RBS" in metrics:
                brier_score = brier_score_loss(label, out_sm[:, 1])
                metric_dict["RBS"] = np.sqrt(brier_score)

            # if "ACE" in metrics:
            #     ace = AdaptiveCalibrationError()
            #     ace.update(out_sm[:, 1], label)
            #     ace = ace.compute()
            #     metric_dict["ACE"] = ace.item()

            results.append(metric_dict)

    res_df = pd.DataFrame(results)

    to_keep = [col for col in list(res_df.columns) if col not in metrics]

    melted_df = res_df.melt(id_vars=to_keep, var_name="metric", value_name="value")

    # sns.set_context(context=None, font_scale=1, rc=insert_rc_args(rc_args))
    # sns.set_style("whitegrid")
    sns.set_palette(color_palette)
    grid = sns.catplot(
        kind="box",
        x=plot_x,
        y=plot_y,
        hue=plot_hue,
        col=plot_col,
        col_wrap=min(3, len(metrics)),
        data=melted_df,
        # estimator="mean",
        # errorbar="sd",
        showfliers=False,
        sharey=False,
        height=4,
        aspect=1,
        legend_out=True,
    )
    grid.figure.set_figwidth(10.5)
    # grid.figure.set_figheight(4)
    # grid.set_titles("{col_name}")
    grid.set_axis_labels("", "Metric Value")
    if custom_legend:
        grid.legend.set_title("Aggregation Method")
        # sns.move_legend(
        #     grid,
        #     "lower center",
        #     bbox_to_anchor=(0.52, -0.2),
        #     ncol=3,
        #     title="Aggregation Method",
        #     frameon=True,
        # )
    # grid.legend.set_title("Method")
    ax1 = grid.axes[0]
    ax1.set_title(r"AUROC $\uparrow$")
    # ax1.set_xlabel(None)
    # ax1.set_ylabel("Value")
    ax2 = grid.axes[1]
    ax2.set_title(r"ECE $\downarrow$")
    # ax2.set_xlabel(None)
    # ax3 = grid.axes[2]
    # ax3.set_title("ECE")
    # ax3.set_xlabel(None)

    grouped_results = res_df.groupby(by=["split", "method"])

    mean_var = grouped_results.agg(lambda x: (np.mean(x), np.std(x)))

    def latex_meanvar_formatter(x):
        mean, var = x
        return f"${mean:.3f} \\pm {var:.3f}$"

    if compute_significance:
        raise RuntimeError("Not correctly implemented")

    if tabname is not None:
        with open(tabname, "w") as f:
            f.write(mean_var.to_latex(formatters=[latex_meanvar_formatter] * len(mean_var.columns), escape=False))

    if figname is not None:
        plt.savefig(figname, bbox_inches="tight")

    return grid


def build_attention_img(
    attn_w: torch.Tensor,
    coords: torch.Tensor,
    scaling_factor: int,
    img_height: int,
    img_width: int,
    patch_size: int = 256,
) -> np.ndarray:
    assert attn_w.shape[0] == coords.shape[0]

    img = np.zeros((img_height, img_width))
    attn_w = attn_w.numpy()
    coords = coords.numpy()

    coords = (coords / scaling_factor).astype("int")
    patch_size = int(patch_size / scaling_factor)

    for i in range(len(attn_w)):
        x, y = coords[i]
        img[y : y + patch_size, x : x + patch_size] = attn_w[i]

    img = (img - np.min(attn_w)) / (np.max(attn_w) - np.min(attn_w))  # Scale attention weights to range [0, 1]

    return img


def overlay_attention(thumbnail_img: np.ndarray, attn_img: np.ndarray):
    fig, ax = plt.subplots(1, 1)
    im = ax.imshow(thumbnail_img, cmap="gray")
    my_cmap = plt.cm.viridis
    my_cmap.set_under("k", alpha=0)
    my_cmap.set_over("k", alpha=0)
    im = ax.imshow(attn_img, cmap=my_cmap, interpolation="none", clim=(0.0, 1.0))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cax=cax)


if __name__ == "__main__":
    RUN_DIR = Path(os.environ["EXPERIMENT_LOCATION"]) / "cluster"
    CONFIG_PATH = "logs/config.yaml"
    SAVE_PATH = Path.cwd().parent / "figures"

    runs = [
        *create_runs_from_folder("slide-calib_256/mco-clam", run_dir=RUN_DIR, config_path=CONFIG_PATH, name="256-CLAM"),
        *create_runs_from_folder(
            "slide-calib_256/mco-transformer", run_dir=RUN_DIR, config_path=CONFIG_PATH, name="256-Transformer"
        ),
        *create_runs_from_folder("slide-calib_256/mco-gnn", run_dir=RUN_DIR, config_path=CONFIG_PATH, name="256-GNN"),
        *create_runs_from_folder("slide-calib_512/mco-clam", run_dir=RUN_DIR, config_path=CONFIG_PATH, name="512-CLAM"),
        *create_runs_from_folder(
            "slide-calib_512/mco-transformer", run_dir=RUN_DIR, config_path=CONFIG_PATH, name="512-Transformer"
        ),
        *create_runs_from_folder("slide-calib_512/mco-gnn", run_dir=RUN_DIR, config_path=CONFIG_PATH, name="512-GNN"),
    ]
    print("Total number of runs: ", len(runs))

    run = runs[20]
    preds, datamodule, dataset = run.get_preds_and_dataset(split="test_id")
    # for i in range(len(dataset)):
    #
    #     if data_dict["coords"].shape[0] < attention.shape[1]:
    #         break
    # No slide has less than 5000 features..
    slide_idx = 150
    attention = preds["attention"][slide_idx]
    indices = preds["indices"][slide_idx]
    data_dict = dataset.get_dict(slide_idx)
    img, info_dict = dataset.get_thumbnail(slide_idx)
    label = preds["label"][slide_idx]
    Y_prob = preds["softmax"][slide_idx]
    print("Slidename: ", data_dict["slidename"])
    print("Label: %d, Label 1 Prob: %.4f" % (label, Y_prob[1].item()))
    # plt.imshow(ImageOps.grayscale(img), cmap="gray")
    # plot_coords(data_dict["coords"][indices] / info_dict["scaling_factor"], data_dict["slidename"])
    # plt.show()
    thumbnail_img = np.asarray(ImageOps.grayscale(img))
    attn_img = build_attention_img(
        attention,
        data_dict["coords"][indices],
        info_dict["scaling_factor"],
        img_height=img.height,
        img_width=img.width,
        patch_size=512,
    )
    overlay_attention(thumbnail_img, attn_img)
    # plt.imshow(attn_img, cmap="coolwarm")
    plt.show()
