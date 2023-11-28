from typing import Tuple
import random
import math
import torch
from torch import nn, Tensor
import torch.nn.functional as F
import torch.optim as optim

CUSTOM_TRANSFORMER = True

if CUSTOM_TRANSFORMER:
    from .models.transformer import TransformerEncoder
else:
    from torch.nn import TransformerEncoder, TransformerEncoderLayer

from .basemodules import BaseModule


class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + math.cos(math.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor


class TransformerModule(BaseModule):
    def __init__(
        self,
        n_classes: int = 2,
        input_dim: int = 512,
        model_dim: int = 256,
        max_seq_len: int = 1000,
        n_heads: int = 4,
        n_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.0,
        input_dropout: float = 0.0,
        lr: float = 5e-4,
        warmup: float = 100,
        aggregate: str = "mean",
        store_attention: bool = False,
    ):
        """
        Args:
            input_dim: Hidden dimensionality of the input
            model_dim: Hidden dimensionality to use inside the Transformer
            n_classes: Number of classes to predict per sequence element
            n_heads: Number of heads to use in the Multi-Head Attention blocks
            n_layers: Number of encoder blocks to use
            dim_feedforward: The dimension of the feedforward part of one TransformerEncoderLayer
            dropout: Dropout to apply inside the model
            input_dropout: Dropout to apply on the input features
            lr: Learning rate in the optimizer
            warmup: Number of warmup steps. Usually between 50 and 500
        """
        super().__init__(n_classes=n_classes)
        self.save_hyperparameters()

        assert aggregate in ["first", "mean", "cls"], "aggregate must be in [first, mean, cls]"
        self.aggregate = aggregate
        # TODO: Need to re-check performance of training with CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, model_dim))

        self.store_attention = store_attention

        # Input dim -> Model dim
        self.input_net = nn.Sequential(nn.Dropout(input_dropout), nn.Linear(input_dim, model_dim))

        if CUSTOM_TRANSFORMER:
            # Transformer from Tutorial
            self.transformer_encoder = TransformerEncoder(
                num_layers=n_layers,
                input_dim=model_dim,
                dim_feedforward=dim_feedforward,
                num_heads=n_heads,
                dropout=dropout,
            )
        else:
            # Pytorch Transformer
            encoder_layer = TransformerEncoderLayer(
                d_model=model_dim,
                nhead=n_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                norm_first=True,
            )
            self.transformer_encoder = TransformerEncoder(
                encoder_layer=encoder_layer,
                num_layers=n_layers,
            )

        # Output classifier per sequence element
        self.classifier = nn.Sequential(
            nn.Linear(model_dim, n_classes),
        )

    def _prepare_transformer_input(self, x: Tensor) -> Tensor:
        # Pass sequence through feedforward net
        x = self.input_net(x)
        batch_size, seq_len, embed_dim = x.size()

        if self.aggregate == "cls":
            # Add cls token to input sequence
            # Use Tensor.repeat as done in https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial15/Vision_Transformer.html
            cls_tokens = self.cls_token.repeat(batch_size, 1, 1)
            cls_tokens = cls_tokens.type(x.type())  # Convert type of cls_tokens to data type
            x = torch.cat([cls_tokens, x], dim=1)

        # TODO: Explore integration of 2D Postional Encodings

        return x

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_dim, n_feats, feat_dim]``
            mask: Tensor, shape ``[seq_len, seq_len]``

        Returns:
            output Tensor of shape ``[batch_dim, n_classes]``
        """
        x = self._prepare_transformer_input(x)

        # Pass sequence through transformer encoder
        x = self.transformer_encoder(x, mask=mask)

        # Extract representative classification token
        if self.aggregate in ["first", "cls"]:
            x = x[:, 0, :]
        if self.aggregate == "mean":
            x = x.mean(dim=1)

        # Pass features through final classification layer
        x = self.classifier(x)
        return x

    def _sample_indices(self, data: Tensor, seed: int = None) -> Tensor:
        batch_dim, n_feats, feat_dim = data.size()
        # We need to randomly sample features such that the Self-Attention mechanism can handle the sequence length
        if n_feats > self.hparams.max_seq_len:
            # Randomly sample max_seq_len features
            # NOTE: Make sure to use a fixed seed at test-time
            if seed is not None:
                # random.seed(seed)
                # indices = random.sample(range(n_feats), self.hparams.max_seq_len)
                indices = torch.randperm(n_feats, generator=torch.Generator().manual_seed(seed))
                indices = indices[: self.hparams.max_seq_len]
            else:
                indices = torch.randperm(n_feats)[: self.hparams.max_seq_len]
                # indices = random.sample(range(n_feats), self.hparams.max_seq_len)
        else:
            indices = torch.arange(n_feats)
        # data = data[:, indices]
        # data = data[:, :1000]  # Cut sequence length for debugging
        return indices  # All returned items should be torch.Tensor

    def forward_logits(self, features: Tensor, coords: Tensor = None, indices: Tensor = None) -> Tensor:
        if indices is None:
            indices = self._sample_indices(features, seed=42)  # Use fixed seed for temperature scaling
        logits = self.forward(features[:, indices])
        return logits

    def _shared_eval_step(self, batch, batch_idx) -> dict:
        features, coords, label = batch
        indices = self._sample_indices(features)

        logits = self.forward(features[:, indices])  # features.shape : [batch_dim, n_feats, feat_dim]

        loss = F.cross_entropy(logits, label)
        Y_prob = F.softmax(logits, dim=1)
        _, Y_hat = torch.max(Y_prob, dim=1)

        # acc = accuracy(Y_hat, label)
        # acc = float(Y_hat.squeeze() == label.squeeze())
        return {
            "loss": loss,
            "logits": logits,
            "Y_prob": Y_prob,
            "Y_hat": Y_hat,
            "label": label,
            "indices": indices,
        }

    # Method for attention map from transformer tutorial
    @torch.no_grad()
    def get_attention_maps(self, x: Tensor, mask: Tensor = None) -> Tensor:
        """
        Function for extracting the attention matrices of the whole Transformer for a single batch.
        Input arguments same as the forward pass.
        """
        x = self._prepare_transformer_input(x)

        attention_maps = self.transformer_encoder.get_attention_maps(x, mask=mask)
        return attention_maps

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
        assert len(features) == 1, "test_step is implemented for batch_size=1"
        indices = self._sample_indices(features, seed=42)

        logits = self.forward_logits(features, indices)
        Y_prob = F.softmax(logits, dim=1)
        self.test_metrics(Y_prob, label)

        # Multiple options to store attention weights. Currently only storing weights for last layer
        if self.store_attention:
            # Can only store attention weights at the sampled indices
            attn_weights = self.get_attention_maps(features[:, indices])
            attn_weights = torch.squeeze(attn_weights)  # Squeeze out batch dimension
            attn_weights = attn_weights[-1, :, :]  # Only look at the weights of the last layer
            if self.aggregate == "cls":
                # Get attention weights for the cls token, excluding its self-attention
                attn_weights = attn_weights[0, 1:]
            elif self.aggregate == "first":
                # Get attention weights for the first token, including its self-attention
                attn_weights = attn_weights[0, :]
            elif self.aggregate == "mean":
                # Only store self-attention weights
                attn_weights = torch.diagonal(attn_weights, offset=0, dim1=0, dim2=1)
            else:
                raise RuntimeError("Unknown aggregation mode")

            assert attn_weights.size() == indices.size(), "Size of attention_weights and indices must match"

            # Return dict to store in self.predictions
            return {
                "softmax": Y_prob,
                "label": label,
                "attention": attn_weights.cpu(),  # Move weights to CPU to free GPU memory
                "indices": indices,
            }
        else:
            return {"softmax": Y_prob, "label": label}

    def predict_step(self, batch, batch_idx):
        raise NotImplementedError

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr)

        # Apply lr scheduler per step
        lr_scheduler = CosineWarmupScheduler(
            optimizer,
            warmup=self.hparams.warmup,
            max_iters=self.trainer.estimated_stepping_batches,
        )

        return [optimizer], [{"scheduler": lr_scheduler, "interval": "step", "frequency": 1}]
