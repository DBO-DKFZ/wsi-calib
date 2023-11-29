# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from curses import A_ALTCHARSET
from typing import Any, List, Optional

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import Tensor

from torchmetrics.functional.classification.calibration_error import (
    _ce_compute,
    _binary_calibration_error_update,
)
from torchmetrics.metric import Metric
from torchmetrics.utilities.data import dim_zero_cat


class CalibrationError(Metric):
    r"""

    `Computes the Top-label Calibration Error`_
    Three different norms are implemented, each corresponding to variations on the calibration error metric.

    L1 norm (Expected Calibration Error)

    .. math::
        \text{ECE} = \frac{1}{N}\sum_i^N \|(p_i - c_i)\|

    Infinity norm (Maximum Calibration Error)

    .. math::
        \text{RMSCE} =  \max_{i} (p_i - c_i)

    L2 norm (Root Mean Square Calibration Error)

    .. math::
        \text{MCE} = \frac{1}{N}\sum_i^N (p_i - c_i)^2

    Where :math:`p_i` is the top-1 prediction accuracy in bin i
    and :math:`c_i` is the average confidence of predictions in bin i.

    .. note::
        L2-norm debiasing is not yet supported.

    Args:
        n_bins: Number of bins to use when computing probabilites and accuracies.
        norm: Norm used to compare empirical and expected probability bins.
            Defaults to "l1", or Expected Calibration Error.
        debias: Applies debiasing term, only implemented for l2 norm. Defaults to True.
        compute_on_step:  Forward only calls ``update()`` and return None if this is set to False.
        dist_sync_on_step: Synchronize metric state across processes at each ``forward()``
            before returning the value at the step
        process_group: Specify the process group on which synchronization is called.
            default: None (which selects the entire world)
    """
    DISTANCES = {"l1", "l2", "max"}
    higher_is_better = False
    confidences: List[Tensor]
    accuracies: List[Tensor]

    def __init__(
        self,
        n_bins: int = 15,
        norm: str = "l1",
    ):
        super().__init__()

        if norm not in self.DISTANCES:
            raise ValueError(f"Norm {norm} is not supported. Please select from l1, l2, or max. ")

        if not isinstance(n_bins, int) or n_bins <= 0:
            raise ValueError(f"Expected argument `n_bins` to be a int larger than 0 but got {n_bins}")
        self.n_bins = n_bins
        self.register_buffer("bin_boundaries", torch.linspace(0, 1, n_bins + 1))
        self.norm = norm

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
        return _ce_compute(confidences, accuracies, self.bin_boundaries, norm=self.norm)

    def plot_reliability_diagram(self, custom_ax=None, title=None, show_legend=True):
        ece = self.compute()

        confidences = dim_zero_cat(self.confidences)
        accuracies = dim_zero_cat(self.accuracies)
        bin_boundaries = self.bin_boundaries

        conf_bin = torch.zeros_like(bin_boundaries)
        acc_bin = torch.zeros_like(bin_boundaries)
        prop_bin = torch.zeros_like(bin_boundaries)
        for i, (bin_lower, bin_upper) in enumerate(zip(bin_boundaries[:-1], bin_boundaries[1:])):
            # Calculated confidence and accuracy in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                acc_bin[i] = accuracies[in_bin].float().mean()
                conf_bin[i] = confidences[in_bin].mean()
                prop_bin[i] = prop_in_bin
        # Shrink array to n_bins
        acc_bin = acc_bin[:-1]
        conf_bin = conf_bin[:-1]
        prop_bin = prop_bin[:-1]

        # Convert relevant tensors to numpy arrays
        bin_boundaries = bin_boundaries.cpu().numpy()
        acc_bin = acc_bin.cpu().numpy()
        conf_bin = conf_bin.cpu().numpy()
        prop_bin = prop_bin.cpu().numpy()
        gap = conf_bin - acc_bin
        pos_gap = np.zeros(len(gap))
        neg_gap = np.zeros(len(gap))
        for i, entry in enumerate(gap):
            if entry > 0:
                pos_gap[i] = entry
            else:
                neg_gap[i] = entry

        width = 1.0 / self.n_bins * 0.9
        x_loc = bin_boundaries + 1 / (2 * self.n_bins)
        x_loc = x_loc[:-1]

        # Plot reliability diagram
        if not custom_ax:
            fig, ax = plt.subplots()
        else:
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

        # Plot adjustments
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Confidence")

        if show_legend:
            ax.legend(loc="upper left")
            ax.set_ylabel("Proportion")

        if title is not None:
            ax.set_title(title + "\nECE: %.4f" % ece, fontsize="medium")
        else:
            ax.set_title("ECE: %.4f" % ece, fontsize="medium")

        return ax
