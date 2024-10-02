import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Model
from torchinfo import summary
import math


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.up_layer = nn.Linear(config.n_embed, 4 * config.n_embed)
        self.fc = nn.GELU()
        self.down_layer = nn.Linear(4 * config.n_embed, config.n_embed)
        self.mlp_dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.up_layer(x)
        x = self.fc(x)
        x = self.down_layer(x)
        x = self.mlp_dropout(x)
        return x


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 输入维度[batch, sequence, embed]
        assert config.n_embed % config.n_head == 0
        self.n_embed = config.n_embed
        self.n_head = config.n_head
        self.c_attn = nn.Linear(self.n_embed, 3 * self.n_embed)
        self.c_proj = nn.Linear(self.n_embed, self.n_embed)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        # mask
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )

    def forward(self, x):
        # bacth size, sequence length, embedding size
        B, T, C = x.size()
        # q k v size: B T n_embed
        q, k, v = self.c_attn(x).split(self.n_embed, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # B n_head T hs
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        attn_weight = (q @ k.transpose(-2, -1)) * (
            1.0 / math.sqrt(k.size(-1))
        )  # B n_head T T (attn_weight)ij = qi * kj 对每行做softmax

        attn_weight = attn_weight.masked_fill_(self.mask[:, :, :T, :T] == 0, float("-inf"))
        attn_weight = F.softmax(attn_weight, dim=-1)
        attn_weight = self.attn_dropout(attn_weight)
        attn = attn_weight @ v  # B n_head T hs
        attn = attn.transpose(1, 2).contiguous().view(B, T, C)
        attn = self.c_proj(attn)
        attn = self.resid_dropout(attn)

        return attn


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(config.n_embed, elementwise_affine=config.ln_affine)
        self.attn = CausalSelfAttention(config)
        self.layer_norm_2 = nn.LayerNorm(config.n_embed, elementwise_affine=config.ln_affine)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.layer_norm_1(x))
        x = x + self.mlp(self.layer_norm_2(x))
        return x


class GPT2Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Embedding()

    def forward(self, x):
        pass


class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304  # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embed: int = 768
    dropout: float = 0.1
    bias: bool = True
    ln_affine: bool = True


if __name__ == "__main__":
    gptconfig = GPTConfig()
    gpt_block = Block(gptconfig)
    print(summary(gpt_block))
