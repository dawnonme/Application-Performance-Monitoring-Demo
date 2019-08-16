import torch.nn as nn
import torch.nn.functional as F


class APMNetwork(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(APMNetwork, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, out_dim),
        )

    def forward(self, x):
        return F.softmax(self.model(x))