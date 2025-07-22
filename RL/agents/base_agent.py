import torch
import torch.nn as nn
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Optional, Any
import logging

class BaseAgent(ABC):
    """Abstract base class for all RL agents"""
    
    def __init__(self, 
                 state_dim: int,
                 action_dim: int,
                 config: Dict[str, Any],
                 device: str = 'cuda'):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Training statistics
        self.training_steps = 0
        self.episodes = 0
        
        # Networks will be initialized in subclasses
        self.networks = {}
        self.optimizers = {}
        self.target_networks = {}
        
    @abstractmethod
    def select_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Select action given state"""
        pass
    
    @abstractmethod
    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Update agent with batch of experiences"""
        pass
    
    @abstractmethod
    def save(self, path: str):
        """Save agent parameters"""
        pass
    
    @abstractmethod
    def load(self, path: str):
        """Load agent parameters"""
        pass
    
    def to_tensor(self, x: np.ndarray) -> torch.Tensor:
        """Convert numpy array to tensor"""
        return torch.FloatTensor(x).to(self.device)
    
    def soft_update(self, target: nn.Module, source: nn.Module, tau: float):
        """Polyak averaging for target networks"""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
    
    def get_statistics(self) -> Dict[str, float]:
        """Get training statistics"""
        return {
            'training_steps': self.training_steps,
            'episodes': self.episodes
        }
