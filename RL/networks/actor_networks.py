import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
from typing import List, Tuple, Optional

class SquashedGaussianActor(nn.Module):
    """
    Gaussian policy with tanh squashing for bounded actions
    Used in SAC and similar algorithms
    """
    
    def __init__(self, 
                 state_dim: int, 
                 action_dim: int, 
                 hidden_dims: List[int] = [256, 256],
                 log_std_min: float = -20,
                 log_std_max: float = 2,
                 epsilon: float = 1e-6):
        super().__init__()
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.epsilon = epsilon
        
        # Build network
        layers = []
        input_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.LayerNorm(hidden_dim))
            input_dim = hidden_dim
        
        self.shared_net = nn.Sequential(*layers)
        
        # Output layers
        self.mean_layer = nn.Linear(input_dim, action_dim)
        self.log_std_layer = nn.Linear(input_dim, action_dim)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning mean and log_std"""
        features = self.shared_net(state)
        mean = self.mean_layer(features)
        log_std = self.log_std_layer(features)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        return mean, log_std
    
    def sample(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample action with reparameterization trick"""
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        # Sample from Gaussian
        normal = Normal(mean, std)
        x = normal.rsample()  # Reparameterization trick
        
        # Apply tanh squashing
        action = torch.tanh(x)
        
        # Calculate log probability with correction for tanh
        log_prob = normal.log_prob(x)
        log_prob -= torch.log(1 - action.pow(2) + self.epsilon)
        log_prob = log_prob.sum(-1, keepdim=True)
        
        # Also return deterministic action
        mean_action = torch.tanh(mean)
        
        return action, log_prob, mean_action

class DeterministicActor(nn.Module):
    """
    Deterministic policy network for DDPG/TD3
    """
    
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 hidden_dims: List[int] = [256, 256],
                 max_action: float = 1.0):
        super().__init__()
        
        self.max_action = max_action
        
        # Build network
        layers = []
        input_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.LayerNorm(hidden_dim))
            input_dim = hidden_dim
        
        layers.append(nn.Linear(input_dim, action_dim))
        layers.append(nn.Tanh())
        
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
        return self.max_action * self.net(state)
