import math
import torch
from torch import nn, Tensor
import torch.nn.functional as F
import torch.optim as optim

import torch_geometric.data as geom_data
import torch_geometric.nn as geom_nn
import pytorch_lightning as pl

# Use absolute imports according to https://peps.python.org/pep-0008/#imports
from src.pl_modules.basemodules import BaseModule
from src.pl_modules.models.gnn import GraphGNNModel


class GraphLevelGNN(BaseModule):
    def __init__(
        self,
        input_dim: int,
        c_in: int,
        c_hidden: int,
        c_out: int,
        layer_name: str = "GAT",
        num_layers: int = 3,
        dp_in: float = 0.1,
        dp_model: float = 0.1,
        dp_out: float = 0.1,
        lr: float = 1e-2,
        weight_decay: float = 0.0,
        **model_kwargs,
    ):
        super().__init__(n_classes=c_out)
        # Saving hyperparameters
        self.save_hyperparameters()

        # Input dim -> Model dim
        self.input_net = nn.Sequential(nn.Dropout(dp_in), nn.Linear(input_dim, c_in))

        self.model = GraphGNNModel(
            c_in=c_in,
            c_hidden=c_hidden,
            c_out=c_out,
            layer_name=layer_name,
            num_layers=num_layers,
            dp_rate=dp_model,
            dp_rate_linear=dp_out,
            **model_kwargs,
        )
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, features: Tensor, coords: Tensor) -> Tensor:
        # Build graph representation for HistoSlides Dataset
        features = features.squeeze()  # Squeeze out batch dimension
        coords = coords.squeeze()
        assert len(coords.shape) == 2, "Method only implemented for batch_size = 1"
        coords = torch.asarray(coords, dtype=torch.float32)
        patch_size = coords[1, 0] - coords[0, 0]
        patch_size = patch_size.item()
        upper_bound = math.ceil(math.sqrt(patch_size**2 + patch_size**2) / 100) * 100
        # Build edge_index and create geom_data Dataset
        edge_index = geom_nn.radius_graph(
            x=coords,
            r=upper_bound,  # Radius r needs to be greater than sqrt(patch_size^2 + patch_size^2) to include 9 neighbors for each coordinate
        )
        data = geom_data.Data(x=features, edge_index=edge_index)

        # We are processing one graph at a time so data.batch returns None
        x, edge_index, batch_idx = data.x, data.edge_index, data.batch
        x = self.input_net(x)  # Reduce feature size from input_dim -> c_in
        x = self.model(x, edge_index, batch_idx)
        logits = x.squeeze(dim=-1)

        return logits

    def forward_logits(self, features: Tensor, coords: Tensor) -> Tensor:
        logits = self.forward(features, coords)
        return logits

    def _shared_eval_step(self, batch, batch_idx) -> dict:
        features, coords, label = batch
        logits = self.forward(features, coords)

        loss = self.criterion(logits, label)
        Y_prob = F.softmax(logits, dim=1)
        _, Y_hat = torch.max(Y_prob, dim=1)

        return {"loss": loss, "Y_prob": Y_prob, "label": label}

    def training_step(self, batch, batch_idx):
        eval_dict = self._shared_eval_step(batch, batch_idx)
        loss = eval_dict["loss"]
        self.train_metrics(eval_dict["Y_prob"], eval_dict["label"])  # Calls update
        metrics = {
            "step": float(self.current_epoch),  # Overwrite step to plot epochs on x-axis
            "train_loss": eval_dict["loss"],
        }
        self.log_dict(metrics, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        eval_dict = self._shared_eval_step(batch, batch_idx)
        self.val_metrics(eval_dict["Y_prob"], eval_dict["label"])
        metrics = {
            "step": float(self.current_epoch),
            "val_loss": eval_dict["loss"],
        }
        self.log_dict(metrics, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        features, coords, label = batch
        logits = self.forward_logits(features, coords)
        Y_prob = F.softmax(logits, dim=1)
        self.test_metrics(Y_prob, label)
        # Return dict to store in self.predictions
        return {"softmax": Y_prob, "label": label}

    def predict_step(self, batch, batch_idx):
        raise NotImplementedError

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )  # High lr because of small dataset and small model
        return optimizer
