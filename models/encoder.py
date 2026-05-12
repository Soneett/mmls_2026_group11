import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def _sparse_mm_safe(A_norm, x):

    if A_norm.is_cuda:

        with torch.amp.autocast("cuda", enabled=False):

            A32 = A_norm.coalesce().to(
                device=x.device,
                dtype=torch.float32
            )

            x32 = x.to(dtype=torch.float32)

            out = torch.sparse.mm(A32, x32)

        return out.to(dtype=x.dtype)

    return torch.sparse.mm(A_norm, x)


class SimpleGCNEncoder(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, n_layers, dropout, use_grad_checkpointing=False):
        super().__init__()

        self.dropout = dropout
        self.use_grad_checkpointing = use_grad_checkpointing

        self.lins = nn.ModuleList()
        self.bns = nn.ModuleList()

        if n_layers == 1:
            self.lins.append(nn.Linear(in_dim, out_dim))
        else:
            self.lins.append(nn.Linear(in_dim, hid_dim))
            self.bns.append(nn.BatchNorm1d(hid_dim))

            for _ in range(n_layers - 2):
                self.lins.append(nn.Linear(hid_dim, hid_dim))
                self.bns.append(nn.BatchNorm1d(hid_dim))

            self.lins.append(nn.Linear(hid_dim, out_dim))

    def _hidden_block(self, li, A_norm, h):
        h = _sparse_mm_safe(A_norm, self.lins[li](h))
        h = self.bns[li](h)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        return h

    def forward(self, A_norm, x):
        h = x

        for li in range(len(self.lins) - 1):
            if self.use_grad_checkpointing and self.training:
                def layer_fn(h_):
                    return self._hidden_block(li, A_norm, h_)

                h = checkpoint(layer_fn, h, use_reentrant=False)
            else:
                h = self._hidden_block(li, A_norm, h)

        h = _sparse_mm_safe(A_norm, self.lins[-1](h))
        return h