import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

class QNetwork(nn.Module):
    """
    Q-network for continuous actions
    """
    
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 hidden_dims: List[int] = [256, 256]):
        super().__init__()
        
        # Build network
        layers = []
        input_dim = state_dim + action_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.LayerNorm(hidden_dim))
            input_dim = hidden_dim
        
        layers.append(nn.Linear(input_dim, 1))
        
        self.net = nn.Sequential(*layers)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        x = torch.cat([state, action], dim=-1)
        return self.net(x)

class TwinQNetwork(nn.Module):
    """
    Twin Q-networks for algorithms like SAC and TD3
    """
    
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 hidden_dims: List[int] = [256, 256]):
        super().__init__()
        
        self.q1 = QNetwork(state_dim, action_dim, hidden_dims)
        self.q2 = QNetwork(state_dim, action_dim, hidden_dims)
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for both Q-networks"""
        q1 = self.q1(state, action)
        q2 = self.q2(state, action)
        return q1, q2

class ValueNetwork(nn.Module):
    """
    State value function V(s)
    """
    
    def __init__(self,
                 state_dim: int,
                 hidden_dims: List[int] = [256, 256]):
        super().__init__()
        
        # Build network
        layers = []
        input_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.LayerNorm(hidden_dim))
            input_dim = hidden_dim
        
        layers.append(nn.Linear(input_dim, 1))
        
        self.net = nn.Sequential(*layers)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        return self.net(state)
