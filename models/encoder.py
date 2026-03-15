import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleGCNEncoder(nn.Module):

    def __init__(self, in_dim, hid_dim, out_dim, n_layers, dropout):

        super().__init__()

        self.dropout = dropout

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

    def forward(self, A_norm, x):

        h = x

        for li in range(len(self.lins) - 1):

            h = torch.sparse.mm(A_norm, self.lins[li](h))

            h = self.bns[li](h)

            h = F.relu(h)

            h = F.dropout(h, p=self.dropout, training=self.training)

        h = torch.sparse.mm(A_norm, self.lins[-1](h))

        return h
