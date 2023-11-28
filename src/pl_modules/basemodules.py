from typing import Any
from pytorch_lightning import LightningModule
import torchmetrics
from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassAccuracy


class BaseModule(LightningModule):
    def __init__(
        self,
        n_classes: int,
    ):
        super().__init__()
        # Initialize metrics
        metrics = MetricCollection(
            {
                "acc": MulticlassAccuracy(num_classes=n_classes, average="micro"),
                "b_acc": MulticlassAccuracy(num_classes=n_classes, average="macro"),
            }
        )
        self.train_metrics = metrics.clone(prefix="train_")
        self.val_metrics = metrics.clone(prefix="val_")
        self.test_metrics = metrics.clone(prefix="test_")

        # Attributes to store predictions
        self.metric_suffix = None
        self.predictions = []

    def training_epoch_end(self, outs):
        metrics = {
            # "step": float(self.current_epoch),  # Do not overwrite step here
            "train_acc": self.train_metrics["acc"],
        }
        self.log_dict(metrics)

    def validation_epoch_end(self, outs):
        self.log_dict(self.val_metrics)

    def test_epoch_end(self, outs):
        self.log_dict(self.test_metrics)
        self.predictions = outs
