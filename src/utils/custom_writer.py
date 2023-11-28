from typing import Any, Sequence
import os
import torch

from pytorch_lightning import LightningModule
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import BasePredictionWriter


def transform_batch_outputs(outs):
    if not isinstance(outs[0], list):
        outs = [outs]

    d_idcs = list(range(len(outs)))

    # Store predictions as module attribute
    results = {d_idx: {} for d_idx in d_idcs}

    for d_idx in d_idcs:
        relevant_outs = outs[d_idx]

        rel_keys = set(relevant_outs[0].keys())
        rel_keys.discard("d_idx")

        for key in rel_keys:
            if key in ["attention", "indices"]:
                # Use list of tensors since tensors have different shapes
                results[d_idx][key] = [out[key].cpu() for out in relevant_outs]
            else:
                results[d_idx][key] = torch.cat([out[key].cpu() for out in relevant_outs])

    return results


def construct_metric_suffix(l_model: LightningModule, d_idx: int, split: str) -> str:
    metric_suffix = None
    if hasattr(l_model, "metric_suffix"):
        metric_suffixes = l_model.metric_suffix
        if isinstance(metric_suffixes, int) or isinstance(metric_suffixes, str):
            metric_suffix = metric_suffixes
        elif isinstance(metric_suffixes, dict):
            if split in metric_suffixes:
                metric_suffixes = metric_suffixes[split]

        if isinstance(metric_suffixes, dict):
            metric_suffix = metric_suffixes.get(d_idx, None)
        elif isinstance(metric_suffixes, list):
            metric_suffix = metric_suffixes[d_idx]

    if metric_suffix is None:
        metric_suffix = str(d_idx) if d_idx != 0 else ""
    elif metric_suffix == "":
        pass
    else:
        metric_suffix = "_" + metric_suffix

    return metric_suffix


class CustomWriter(BasePredictionWriter):
    def __init__(self, output_dir: str, write_interval: str, name: str = "out"):
        super().__init__(write_interval)
        self.output_dir = output_dir
        self.name = name

    def write_on_batch_end(
        self,
        trainer,
        pl_module: LightningModule,
        prediction: Any,
        batch_indices: list[int],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ):
        torch.save(prediction, os.path.join(self.output_dir, dataloader_idx, f"{batch_idx}.pt"))

    def write_on_epoch_end(
        self, trainer, pl_module: LightningModule, predictions: dict[Any], batch_indices: list[Any] = []
    ):
        predictions = transform_batch_outputs(predictions)
        d_idcs = list(predictions.keys())

        # Save the predictions for each dataset separately.
        for d_idx in d_idcs:
            # We can construct the name from a) The dataset name attribute
            #                                b) The pl_module.metric_suffix attribute
            #                                c) the d_idx
            name = None
            # if hasattr(trainer, "test_dataloaders") and len(trainer.test_dataloaders) == max(d_idcs) + 1:
            #     if trainer.test_dataloaders[d_idx].dataset.name is not None:
            #         name = trainer.test_dataloaders[d_idx].dataset.name

            # Option b) and c)
            if name is None:
                name_suffix = construct_metric_suffix(pl_module, d_idx, "test")
                if name_suffix == "":
                    name = "test"
                else:
                    name = "test_" + name_suffix[1:]

            torch.save(predictions[d_idx], os.path.join(self.output_dir, f"{name}_{self.name}.preds"))

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        predictions = pl_module.predictions
        # epoch_batch_indices = trainer.test_loop.epoch_batch_indices
        self.write_on_epoch_end(trainer, pl_module, predictions)

    def on_predict_epoch_end(self, trainer: Trainer, pl_module: LightningModule, outputs: Sequence[Any]) -> None:
        return
        # predictions = pl_module.predictions
        # epoch_batch_indices = trainer.test_loop.epoch_batch_indices
        # self.write_on_epoch_end(trainer, pl_module, predictions, batch_indices=[])
