import torch
from torch import nn, Tensor
import torch.nn.functional as F
import torch.optim as optim

from .basemodules import BaseModule
from .models.clam import CLAM_SB, CLAM_MB
from topk.svm import SmoothTop1SVM


class ClamModule(BaseModule):
    def __init__(
        self,
        n_classes: int = 2,
        feat_size: int = 1024,
        lr: float = 2e-4,
        weight_decay: float = 1e-5,
        bag_weight: float = 0.7,
    ):
        super().__init__(n_classes=n_classes)
        self.save_hyperparameters()
        # instance_loss_fn is initialized on the cpu and later needs to be moved to device
        instance_loss_fn = SmoothTop1SVM(n_classes=n_classes)
        self.model = CLAM_SB(
            gate=True,
            size_arg="small",
            feat_size=feat_size,
            dropout=True,
            k_sample=8,
            n_classes=n_classes,
            instance_loss_fn=instance_loss_fn.cuda(),  # TODO: Need to find generic way to move loss_fn to device
            subtyping=False,
        )

    def forward_logits(self, features: Tensor, coords: Tensor = None) -> Tensor:
        assert len(features) == 1, "Method developed for batch size 1"
        features = torch.squeeze(features)  # Squeeze out batch dimension
        logits, Y_prob, Y_hat, _, instance_dict = self.model(features)
        return logits

    def _shared_eval_step(self, batch, batch_idx) -> dict:
        features, coords, label = batch
        assert len(features) == 1, "Method developed for batch size 1"
        features = torch.squeeze(features)  # Squeeze out batch dimension
        logits, Y_prob, Y_hat, _, instance_dict = self.model(features, label=label, instance_eval=True)
        Y_hat = Y_hat.reshape(-1)  # Remove additional dimension from Y_hat
        assert Y_hat.shape == label.shape

        bag_loss = F.cross_entropy(logits, label)
        instance_loss = instance_dict["instance_loss"]
        total_loss = self.hparams.bag_weight * bag_loss + (1 - self.hparams.bag_weight) * instance_loss

        # acc = accuracy(Y_hat, label)
        # acc = float(Y_hat.squeeze() == label.squeeze())
        return {
            "total_loss": total_loss,
            "bag_loss": bag_loss,
            "instance_loss": instance_loss,
            "logits": logits,
            "Y_prob": Y_prob,
            "Y_hat": Y_hat,
            "label": label,
        }

    def training_step(self, batch, batch_idx):
        eval_dict = self._shared_eval_step(batch, batch_idx)
        loss = eval_dict["total_loss"]
        self.train_metrics(eval_dict["Y_prob"], eval_dict["label"])  # Calls update
        metrics = {
            "step": float(self.current_epoch),  # Overwrite step to plot epochs on x-axis
            "train_loss": eval_dict["total_loss"],
            "train_bag_loss": eval_dict["bag_loss"],
            "train_instance_loss": eval_dict["instance_loss"],
        }
        self.log_dict(metrics, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        eval_dict = self._shared_eval_step(batch, batch_idx)
        self.val_metrics(eval_dict["Y_prob"], eval_dict["label"])
        metrics = {
            "step": float(self.current_epoch),
            "val_loss": eval_dict["total_loss"],
        }
        self.log_dict(metrics, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        features, coords, label = batch
        logits = self.forward_logits(features)
        Y_prob = F.softmax(logits, dim=1)
        self.test_metrics(Y_prob, label)
        # Return dict to store in self.predictions
        return {"softmax": Y_prob, "label": label}

    def predict_step(self, batch, batch_idx):
        raise NotImplementedError

    def configure_optimizers(self):
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        return optimizer
