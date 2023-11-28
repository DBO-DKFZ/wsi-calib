from typing import Any, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import Tensor, FloatTensor

from torchmetrics.functional.classification.calibration_error import (
    _ce_compute,
    _binary_calibration_error_update,
)
from torchmetrics.metric import Metric
from torchmetrics.utilities.data import dim_zero_cat


def adaptive_binning(confidences: FloatTensor, accuracies: FloatTensor) -> Tensor:
    """Computes the adaptive calibration error given the confidence and accuracy values.
    Implementation from https://github.com/yding5/AdaptiveBinning/blob/master/AdaptiveBinning.py

    Args:
        confidences (FloatTensor): The confidence (i.e. predicted prob) of the top1 prediction.
        accuracies (FloatTensor): 1.0 if the top-1 prediction was correct, 0.0 otherwise.

    Returns:
        Tensor: Adaptive calibration error scalar.
    """
    infer_results = torch.hstack((confidences[:, None], accuracies[:, None]))
    infer_results = infer_results.tolist()

    # Intialize.
    infer_results.sort(key=lambda x: x[0], reverse=True)
    n_total_sample = len(infer_results)

    assert infer_results[0][0] <= 1 and infer_results[1][0] >= 0, "Confidence score should be in [0,1]"

    z = 1.645
    num = [0 for i in range(n_total_sample)]
    final_num = [0 for i in range(n_total_sample)]
    correct = [0 for i in range(n_total_sample)]
    confidence = [0 for i in range(n_total_sample)]
    cof_min = [1 for i in range(n_total_sample)]
    cof_max = [0 for i in range(n_total_sample)]
    accuracy = [0 for i in range(n_total_sample)]

    ind = 0
    target_number_samples = float("inf")

    # Traverse all samples for a initial binning.
    for i, confindence_correctness in enumerate(infer_results):
        confidence_score = confindence_correctness[0]
        correctness = confindence_correctness[1]
        # Merge the last bin if too small.
        if num[ind] > target_number_samples:
            if (n_total_sample - i) > 40 and cof_min[ind] - infer_results[-1][0] > 0.05:
                ind += 1
                target_number_samples = float("inf")
        num[ind] += 1
        confidence[ind] += confidence_score

        assert correctness in [True, False], "Expect boolean value for correctness!"

        if correctness == True:
            correct[ind] += 1

        cof_min[ind] = min(cof_min[ind], confidence_score)
        cof_max[ind] = max(cof_max[ind], confidence_score)
        # Get target number of samples in the bin.
        if cof_max[ind] == cof_min[ind]:
            target_number_samples = float("inf")
        else:
            target_number_samples = (z / (cof_max[ind] - cof_min[ind])) ** 2 * 0.25

    n_bins = ind + 1

    # Get final binning.
    if target_number_samples - num[ind] > 0:
        needed = target_number_samples - num[ind]
        extract = [0 for i in range(n_bins - 1)]
        final_num[n_bins - 1] = num[n_bins - 1]
        for i in range(n_bins - 1):
            extract[i] = int(needed * num[ind] / n_total_sample)
            final_num[i] = num[i] - extract[i]
            final_num[n_bins - 1] += extract[i]
    else:
        final_num = num
    final_num = final_num[:n_bins]

    # Re-intialize.
    num = [0 for i in range(n_bins)]
    correct = [0 for i in range(n_bins)]
    confidence = [0 for i in range(n_bins)]
    cof_min = [1 for i in range(n_bins)]
    cof_max = [0 for i in range(n_bins)]
    accuracy = [0 for i in range(n_bins)]
    gap = [0 for i in range(n_bins)]
    neg_gap = [0 for i in range(n_bins)]
    # Bar location and width.
    x_location = [0 for i in range(n_bins)]
    width = [0 for i in range(n_bins)]

    # Calculate confidence and accuracy in each bin.
    ind = 0
    for i, confindence_correctness in enumerate(infer_results):
        confidence_score = confindence_correctness[0]
        correctness = confindence_correctness[1]
        num[ind] += 1
        confidence[ind] += confidence_score

        if correctness == True:
            correct[ind] += 1
        cof_min[ind] = min(cof_min[ind], confidence_score)
        cof_max[ind] = max(cof_max[ind], confidence_score)

        if num[ind] == final_num[ind]:
            confidence[ind] = confidence[ind] / num[ind] if num[ind] > 0 else 0
            accuracy[ind] = correct[ind] / num[ind] if num[ind] > 0 else 0
            left = cof_min[ind]
            right = cof_max[ind]
            x_location[ind] = (left + right) / 2
            width[ind] = (right - left) * 0.9
            if confidence[ind] - accuracy[ind] > 0:
                gap[ind] = confidence[ind] - accuracy[ind]
            else:
                neg_gap[ind] = confidence[ind] - accuracy[ind]
            ind += 1

    # Convert lists to torch tensors
    acc_bin = torch.tensor(accuracy)
    conf_bin = torch.tensor(confidence)
    prop_bin = [samples / n_total_sample for samples in final_num]
    prop_bin = torch.tensor(prop_bin)

    return acc_bin, conf_bin, prop_bin, x_location, width


def _aece_compute(acc_bin, conf_bin, prop_bin):
    return torch.sum(torch.abs(acc_bin - conf_bin) * prop_bin)


class AdaptiveCalibrationError(Metric):
    DISTANCES = {"l1", "l2", "max"}
    higher_is_better = False
    confidences: List[Tensor]
    accuracies: List[Tensor]

    def __init__(self):
        super().__init__()

        self.n_bins = 0
        self.acc_bin = torch.Tensor()
        self.conf_bin = torch.Tensor()
        self.prop_bin = torch.Tensor()
        self.bin_loc = []
        self.bin_width = []

        self.add_state("confidences", [], dist_reduce_fx="cat")
        self.add_state("accuracies", [], dist_reduce_fx="cat")

    def update(self, preds: Tensor, target: Tensor) -> None:  # type: ignore
        """Computes top-level confidences and accuracies for the input probabilites and appends them to internal
        state.

        Args:
            preds (Tensor): Model output probabilities.
            target (Tensor): Ground-truth target class labels.
        """
        confidences, accuracies = _binary_calibration_error_update(preds, target)

        self.confidences.append(confidences)
        self.accuracies.append(accuracies)

    def compute(self) -> Tensor:
        """Computes calibration error across all confidences and accuracies.

        Returns:
            Tensor: Calibration error across previously collected examples.
        """

        confidences = dim_zero_cat(self.confidences)
        accuracies = dim_zero_cat(self.accuracies)
        acc_bin, conf_bin, prop_bin, bin_loc, bin_width = adaptive_binning(confidences, accuracies)
        self.n_bins = len(bin_loc)
        self.acc_bin = acc_bin
        self.conf_bin = conf_bin
        self.prop_bin = prop_bin
        self.bin_loc = bin_loc
        self.bin_width = bin_width
        aece = _aece_compute(acc_bin, conf_bin, prop_bin)
        return aece

    def plot_reliability_diagram(self, custom_ax=None, title=None):
        aece = self.compute()

        # Convert relevant tensors to numpy arrays
        acc_bin = self.acc_bin.cpu().numpy()
        conf_bin = self.conf_bin.cpu().numpy()
        prop_bin = self.prop_bin.cpu().numpy()
        gap = conf_bin - acc_bin
        pos_gap = np.zeros(len(gap))
        neg_gap = np.zeros(len(gap))
        for i, entry in enumerate(gap):
            if entry > 0:
                pos_gap[i] = entry
            else:
                neg_gap[i] = entry

        width = self.bin_width
        x_loc = self.bin_loc

        # Plot reliability diagram
        fig, ax = plt.subplots()
        if custom_ax:
            ax = custom_ax
        ax.bar(x_loc, acc_bin, width, label="Accuracy")
        ax.bar(x_loc, pos_gap, width, bottom=acc_bin, label="Pos(Conf-Acc)")
        ax.bar(x_loc, neg_gap, width, bottom=acc_bin, label="Neg(Conf-Acc)")

        ax.plot(
            np.linspace(0, 1.0, 10),
            np.linspace(0, 1.0, 10),
            c="gray",
            linestyle="dashed",
        )

        ax.plot(x_loc, prop_bin, marker="o", color="black", label="Sample proportion")

        ax.legend(loc="upper left")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Accuracy")
        if title is not None:
            ax.set_title(title + "\nAECE: %.4f" % aece)
        else:
            ax.set_title("AECE: %.4f" % aece)

        return fig, ax
