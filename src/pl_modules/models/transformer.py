# Code from https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial6/Transformers_and_MHAttention.html

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

CUSTOM_MHA = True

if not CUSTOM_MHA:
    from torch.nn import MultiheadAttention


def scaled_dot_product(q, k, v, mask=None):
    d_k = q.size()[-1]
    attn_logits = torch.matmul(q, k.transpose(-2, -1))
    attn_logits = attn_logits / math.sqrt(d_k)
    if mask is not None:
        attn_logits = attn_logits.masked_fill(mask == 0, -9e15)
    attention = F.softmax(attn_logits, dim=-1)
    values = torch.matmul(attention, v)
    return values, attention


if CUSTOM_MHA:

    class MultiheadAttention(nn.Module):
        def __init__(self, input_dim, embed_dim, num_heads):
            super().__init__()
            assert embed_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."

            self.embed_dim = embed_dim
            self.num_heads = num_heads
            self.head_dim = embed_dim // num_heads

            # Stack all weight matrices 1...h together for efficiency
            # Note that in many implementations you see "bias=False" which is optional
            self.qkv_proj = nn.Linear(input_dim, 3 * embed_dim)
            self.o_proj = nn.Linear(embed_dim, embed_dim)

            self._reset_parameters()

        def _reset_parameters(self):
            # Original Transformer initialization, see PyTorch documentation
            nn.init.xavier_uniform_(self.qkv_proj.weight)
            self.qkv_proj.bias.data.fill_(0)
            nn.init.xavier_uniform_(self.o_proj.weight)
            self.o_proj.bias.data.fill_(0)

        def forward(self, x, mask=None, return_attention=False):
            batch_size, seq_length, embed_dim = x.size()
            qkv = self.qkv_proj(x)

            # Separate Q, K, V from linear output
            qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3 * self.head_dim)
            qkv = qkv.permute(0, 2, 1, 3)  # [Batch, Head, SeqLen, Dims]
            q, k, v = qkv.chunk(3, dim=-1)

            # Determine value outputs
            values, attention = scaled_dot_product(q, k, v, mask=mask)
            values = values.permute(0, 2, 1, 3)  # [Batch, SeqLen, Head, Dims]
            values = values.reshape(batch_size, seq_length, embed_dim)
            o = self.o_proj(values)

            if return_attention:
                return o, attention
            else:
                return o


class EncoderBlock(nn.Module):
    def __init__(self, input_dim, num_heads, dim_feedforward, dropout=0.0):
        """
        Args:
            input_dim: Dimensionality of the input
            num_heads: Number of heads to use in the attention block
            dim_feedforward: Dimensionality of the hidden layer in the MLP
            dropout: Dropout probability to use in the dropout layers
        """
        super().__init__()

        # Attention layer
        if CUSTOM_MHA:
            self.self_attn = MultiheadAttention(input_dim, input_dim, num_heads)
        else:
            self.self_attn = MultiheadAttention(
                input_dim,
                num_heads,
                batch_first=True,  # Important to set batch_first correctly!
            )

        # Two-layer MLP
        self.linear_net = nn.Sequential(
            nn.Linear(input_dim, dim_feedforward),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
            nn.Linear(dim_feedforward, input_dim),
        )

        # Layers to apply in between the main layers
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Attention part
        if CUSTOM_MHA:
            attn_out, attn_weights = self.self_attn(x, mask=mask, return_attention=True)
        else:
            attn_out, attn_weights = self.self_attn(x, x, x, attn_mask=mask, need_weights=True)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)

        # MLP part
        linear_out = self.linear_net(x)
        x = x + self.dropout(linear_out)
        x = self.norm2(x)

        return x


class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, **block_args):
        super().__init__()
        self.layers = nn.ModuleList([EncoderBlock(**block_args) for _ in range(num_layers)])

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask=mask)
        return x

    def get_attention_maps(self, x, mask=None) -> torch.Tensor:
        attention_maps = []
        for layer in self.layers:
            if CUSTOM_MHA:
                attn_out, attn_weights = layer.self_attn(x, mask=mask, return_attention=True)
                # Average weights across heads, which is the default setting in Pytorch implementation
                attn_weights = torch.mean(attn_weights, dim=1)
            else:
                attn_out, attn_weights = layer.self_attn(x, x, x, attn_mask=mask, need_weights=True)
            # attn_weights has shape [N, L, S], where N is the batch_size, L the target sequence length and S the source sequence length
            attention_maps.append(attn_weights)
            x = layer(x)
        attention_maps = torch.stack(attention_maps)
        # Turn list of tensors into tensor with shape [batch_size, n_layers, target_seq_len, source_seq_len]
        attention_maps = attention_maps.permute(1, 0, 2, 3)
        return attention_maps


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        """
        Args
            d_model: Hidden dimensionality of the input.
            max_len: Maximum length of a sequence to expect.
        """
        super().__init__()

        # Create matrix of [SeqLen, HiddenDim] representing the positional encoding for max_len inputs
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        # register_buffer => Tensor which is not a parameter, but should be part of the modules state.
        # Used for tensors that need to be on the same device as the module.
        # persistent=False tells PyTorch to not add the buffer to the state dict (e.g. when we save the model)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return x


if __name__ == "__main__":
    # 1) Attention
    import pytorch_lightning as pl

    pl.seed_everything(42)

    seq_len, d_k = 3, 2
    q = torch.randn(seq_len, d_k)
    k = torch.randn(seq_len, d_k)
    v = torch.randn(seq_len, d_k)
    values, attention = scaled_dot_product(q, k, v)
    print("Q\n", q)
    print("K\n", k)
    print("V\n", v)
    print("Values\n", values)
    print("Attention\n", attention)

    # 2a) Positional Encoding
    import matplotlib.pyplot as plt

    encod_block = PositionalEncoding(d_model=48, max_len=96)
    pe = encod_block.pe.squeeze().T.cpu().numpy()

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 3))
    pos = ax.imshow(pe, cmap="RdGy", extent=(1, pe.shape[1] + 1, pe.shape[0] + 1, 1))
    fig.colorbar(pos, ax=ax)
    ax.set_xlabel("Position in sequence")
    ax.set_ylabel("Hidden dimension")
    ax.set_title("Positional encoding over hidden dimensions")
    ax.set_xticks([1] + [i * 10 for i in range(1, 1 + pe.shape[1] // 10)])
    ax.set_yticks([1] + [i * 10 for i in range(1, 1 + pe.shape[0] // 10)])
    plt.show()

    # 2b) Positional Encoding
    import seaborn as sns
    import numpy as np

    sns.set_theme()
    fig, ax = plt.subplots(2, 2, figsize=(12, 4))
    ax = [a for a_list in ax for a in a_list]
    for i in range(len(ax)):
        ax[i].plot(np.arange(1, 17), pe[i, :16], color="C%i" % i, marker="o", markersize=6, markeredgecolor="black")
        ax[i].set_title("Encoding in hidden dimension %i" % (i + 1))
        ax[i].set_xlabel("Position in sequence", fontsize=10)
        ax[i].set_ylabel("Positional encoding", fontsize=10)
        ax[i].set_xticks(np.arange(1, 17))
        ax[i].tick_params(axis="both", which="major", labelsize=10)
        ax[i].tick_params(axis="both", which="minor", labelsize=8)
        ax[i].set_ylim(-1.2, 1.2)
    fig.subplots_adjust(hspace=0.8)
    sns.reset_orig()
    plt.show()
