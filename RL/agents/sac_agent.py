import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Dict, Tuple, Optional
from .base_agent import BaseAgent
from ..networks.actor_networks import SquashedGaussianActor
from ..networks.critic_networks import TwinQNetwork

class SACAgent(BaseAgent):
    """
    Soft Actor-Critic agent optimized for noisy financial data
    Includes entropy regularization for exploration
    """
    
    def __init__(self, state_dim: int, action_dim: int, config: Dict, device: str = 'cuda'):
        super().__init__(state_dim, action_dim, config, device)
        
        # Hyperparameters
        self.gamma = config.get('gamma', 0.99)
        self.tau = config.get('tau', 0.005)
        self.alpha = config.get('alpha', 0.2)
        self.automatic_entropy_tuning = config.get('automatic_entropy_tuning', True)
        self.hidden_dims = config.get('hidden_dims', [512, 512, 256])
        self.learning_rate = config.get('learning_rate', 3e-4)
        
        # Networks
        self.actor = SquashedGaussianActor(
            state_dim, action_dim, self.hidden_dims
        ).to(self.device)
        
        self.critic = TwinQNetwork(
            state_dim, action_dim, self.hidden_dims
        ).to(self.device)
        
        self.critic_target = TwinQNetwork(
            state_dim, action_dim, self.hidden_dims
        ).to(self.device)
        
        # Copy target parameters
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Optimizers
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(), lr=self.learning_rate
        )
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(), lr=self.learning_rate
        )
        
        # Automatic entropy tuning
        if self.automatic_entropy_tuning:
            self.target_entropy = -float(action_dim)
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.learning_rate)
            self.alpha = self.log_alpha.exp()
        
    def select_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Select action using current policy"""
        state_tensor = self.to_tensor(state).unsqueeze(0)
        
        with torch.no_grad():
            if deterministic:
                _, _, action = self.actor.sample(state_tensor)
            else:
                action, _, _ = self.actor.sample(state_tensor)
        
        return action.cpu().numpy()[0]
    
    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Update SAC agent"""
        states = batch['states']
        actions = batch['actions']
        rewards = batch['rewards']
        next_states = batch['next_states']
        dones = batch['dones']
        
        # Update critic
        with torch.no_grad():
            next_actions, next_log_probs, _ = self.actor.sample(next_states)
            q1_next, q2_next = self.critic_target(next_states, next_actions)
            q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_probs
            q_target = rewards + self.gamma * (1 - dones) * q_next
        
        q1, q2 = self.critic(states, actions)
        critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()
        
        # Update actor
        actions_new, log_probs, _ = self.actor.sample(states)
        q1_new, q2_new = self.critic(states, actions_new)
        q_new = torch.min(q1_new, q2_new)
        
        actor_loss = (self.alpha * log_probs - q_new).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()
        
        # Update temperature
        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
            
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            
            self.alpha = self.log_alpha.exp()
        else:
            alpha_loss = torch.tensor(0.0)
        
        # Update target networks
        self.soft_update(self.critic_target, self.critic, self.tau)
        
        self.training_steps += 1
        
        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'alpha_loss': alpha_loss.item(),
            'alpha': self.alpha.item() if self.automatic_entropy_tuning else self.alpha,
            'q_value': q_new.mean().item()
        }
    
    def save(self, path: str):
        """Save agent parameters"""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'log_alpha': self.log_alpha if self.automatic_entropy_tuning else None,
            'alpha_optimizer': self.alpha_optimizer.state_dict() if self.automatic_entropy_tuning else None,
            'training_steps': self.training_steps,
            'config': self.config
        }, path)
    
    def load(self, path: str):
        """Load agent parameters"""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        
        if self.automatic_entropy_tuning and checkpoint['log_alpha'] is not None:
            self.log_alpha = checkpoint['log_alpha']
            self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer'])
            self.alpha = self.log_alpha.exp()
        
        self.training_steps = checkpoint['training_steps']
