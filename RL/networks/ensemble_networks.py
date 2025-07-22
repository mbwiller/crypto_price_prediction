import torch
import torch.nn as nn
from typing import List

class MetaNetwork(nn.Module):
    """Simple feed-forward network that outputs weighting coefficients for base agents."""

    def __init__(self, input_dim: int, num_agents: int, hidden_dims: List[int] = None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 128]
        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.LayerNorm(h))
            in_dim = h
        layers.append(nn.Linear(in_dim, num_agents))
        self.net = nn.Sequential(*layers)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
