import torch
import torch.nn as nn


class SimpleMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 25), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(25, 10),         nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(10, 1),
        )
    def forward(self, x):
        return self.net(x)
