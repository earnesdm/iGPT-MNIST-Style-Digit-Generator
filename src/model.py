import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# create device
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
H, W, _ = (28, 28, 1)


def sample_model(model, x):
    model.eval()
    with torch.no_grad():
        for i in range(H*W):
            output = model(x)
            x[:, i, :] = torch.bernoulli(output[:, i, :])
        return x


class MaskedAttention(nn.Module):
    def __init__(self, d_model, heads):
        super().__init__()
        # queries, keys, and values
        self._q = nn.Linear(d_model, d_model)
        self._k = nn.Linear(d_model, d_model)
        self._v = nn.Linear(d_model, d_model)

        # lower diagonal matrix (diagonal is all zeros)
        self.mask = torch.tril(torch.ones((H * W, H * W))).to(device=device)

        self.heads = heads

    def forward(self, x):
        # input has shape (N, H*W, d_model)
        q = self._q(x)
        k = self._k(x)
        v = self._k(x)

        # reshape q, k, and v to have shape (N, heads, H*W, head_size)
        N, H_W, d_model = q.shape
        q = q.view(N, H_W, self.heads, d_model // self.heads).transpose(1, 2)
        k = k.view(N, H_W, self.heads, d_model // self.heads).transpose(1, 2)
        v = v.view(N, H_W, self.heads, d_model // self.heads).transpose(1, 2)

        # TODO: properly add heads
        atten = (q @ k.transpose(2, 3)) / np.sqrt(k.shape[-1])

        # when the mask is 0 we set the logits to a large negative number
        atten = atten.masked_fill(self.mask == 0, -np.inf)
        atten = F.softmax(atten, dim=-1).masked_fill(self.mask == 0, 0)
        # atten = F.softmax(atten - (1 - self.mask) * 1e10, dim=-1)

        out = (atten @ v).transpose(1, 2).contiguous().view(N, H_W, -1)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, d_model, heads):
        super().__init__()
        self._ln1 = nn.LayerNorm(d_model)
        self._ln2 = nn.LayerNorm(d_model)
        self._attn1 = MaskedAttention(d_model, heads)
        self._out = nn.Sequential(
            nn.Linear(in_features=d_model, out_features=d_model * 4),
            nn.GELU(),
            nn.Linear(in_features=d_model * 4, out_features=d_model),
        )

    def forward(self, x):
        x = x + self._attn1(self._ln1(x))
        return x + self._out(self._ln2(x))


class ImageGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.d_model = 128
        self.heads = 4

        # learned positional encoding
        self._pos = nn.Parameter(torch.zeros(1, H * W, self.d_model))

        # input embedding
        self.input_embed = nn.Linear(in_features=1, out_features=self.d_model)

        # transformer blocks
        self.tf1 = TransformerBlock(self.d_model, self.heads)
        self.tf2 = TransformerBlock(self.d_model, self.heads)

        # outputs logits (we only need 1 outputs becasue the pixels are binary)
        self.ln1 = nn.LayerNorm(self.d_model)
        self._out = nn.Linear(in_features=self.d_model, out_features=1)

    def forward(self, x):
        # shift input and add the begining of sequence token
        x = torch.cat((torch.ones((x.shape[0], 1, x.shape[2])).to(device=device) * -1, x[:, :-1, :]), dim=1)
        # print(x.shape)
        # print(x[0])

        # embed the input and add positional encodding
        x = self.input_embed(x) + self._pos

        # transformer blocks
        x = self.tf1(x)
        x = self.tf2(x)

        return F.sigmoid(self._out(self.ln1(x)))
