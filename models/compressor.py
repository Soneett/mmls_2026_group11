import torch.nn as nn

class Compressor(nn.Module):

    def __init__(self, d_in, d_out):

        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(d_in, d_out * 2),
            nn.ReLU(),
            nn.Linear(d_out * 2, d_out),
            nn.LayerNorm(d_out),
        )

    def forward(self, x):

        return self.net(x)
