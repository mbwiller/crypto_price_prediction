import numpy as np
import torch
from typing import Dict, Optional, Tuple

class ReplayBuffer:
    """
    Standard replay buffer for off-policy RL algorithms
    """
    
    def __init__(self, capacity: int, state_dim: Optional[int] = None):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        self.state_dim = state_dim
    
    def add(self, state: np.ndarray, action: np.ndarray, reward: float, 
            next_state: np.ndarray, done: bool):
        """Add transition to buffer"""
        
        transition = {
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done
        }
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.position] = transition
        
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int, device: str = 'cuda') -> Dict[str, torch.Tensor]:
        """Sample batch of transitions"""
        
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        
        for idx in indices:
            transition = self.buffer[idx]
            states.append(transition['state'])
            actions.append(transition['action'])
            rewards.append(transition['reward'])
            next_states.append(transition['next_state'])
            dones.append(transition['done'])
        
        # Convert to tensors
        batch = {
            'states': torch.FloatTensor(np.array(states)).to(device),
            'actions': torch.FloatTensor(np.array(actions)).to(device),
            'rewards': torch.FloatTensor(np.array(rewards)).unsqueeze(1).to(device),
            'next_states': torch.FloatTensor(np.array(next_states)).to(device),
            'dones': torch.FloatTensor(np.array(dones)).unsqueeze(1).to(device)
        }
        
        return batch
    
    def __len__(self):
        return len(self.buffer)
