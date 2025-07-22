import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
from .base_agent import BaseAgent
from .sac_agent import SACAgent
from .ppo_agent import PPOAgent
from .ddpg_agent import DDPGAgent
from .td3_agent import TD3Agent
from .td_mpc2_agent import TDMPC2Agent
from ..networks.ensemble_networks import MetaNetwork

class EnsembleAgent(BaseAgent):
    """
    Hierarchical ensemble agent that combines predictions from multiple base agents
    Implements the meta-RL approach described in the research
    """
    
    def __init__(self, state_dim: int, action_dim: int, config: Dict, device: str = 'cuda'):
        super().__init__(state_dim, action_dim, config, device)
        
        # Configuration
        self.base_agents_config = config.get('base_agents', {
            'sac': {'enabled': True},
            'ppo': {'enabled': True},
            'ddpg': {'enabled': True},
            'td3': {'enabled': True},
            'td_mpc2': {'enabled': True}
        })
        
        # Initialize base agents
        self.base_agents = {}
        self._initialize_base_agents()
        
        # Meta-state dimension: original state + predictions from all base agents
        self.meta_state_dim = state_dim + len(self.base_agents) * action_dim
        
        # Meta-agent (using SAC for stability)
        meta_config = config.get('meta_agent_config', {})
        self.meta_agent = SACAgent(
            self.meta_state_dim, 
            action_dim, 
            meta_config, 
            device
        )
        
        # Meta-network for adaptive weighting
        self.meta_network = MetaNetwork(
            self.meta_state_dim,
            len(self.base_agents),
            hidden_dims=config.get('meta_hidden_dims', [256, 256])
        ).to(self.device)
        
        self.meta_optimizer = torch.optim.Adam(
            self.meta_network.parameters(),
            lr=config.get('meta_learning_rate', 1e-4)
        )
        
        # Training mode flags
        self.train_base_agents = config.get('train_base_agents', True)
        self.train_meta_agent = config.get('train_meta_agent', True)
        
    def _initialize_base_agents(self):
        """Initialize all enabled base agents"""
        
        if self.base_agents_config['sac']['enabled']:
            self.base_agents['sac'] = SACAgent(
                self.state_dim, 
                self.action_dim,
                self.base_agents_config['sac'],
                self.device
            )
        
        if self.base_agents_config['ppo']['enabled']:
            from .ppo_agent import PPOAgent
            self.base_agents['ppo'] = PPOAgent(
                self.state_dim,
                self.action_dim,
                self.base_agents_config['ppo'],
                self.device
            )
        
        if self.base_agents_config['ddpg']['enabled']:
            from .ddpg_agent import DDPGAgent
            self.base_agents['ddpg'] = DDPGAgent(
                self.state_dim,
                self.action_dim,
                self.base_agents_config['ddpg'],
                self.device
            )
        
        if self.base_agents_config['td3']['enabled']:
            from .td3_agent import TD3Agent
            self.base_agents['td3'] = TD3Agent(
                self.state_dim,
                self.action_dim,
                self.base_agents_config['td3'],
                self.device
            )
        
        if self.base_agents_config['td_mpc2']['enabled']:
            from .td_mpc2_agent import TDMPC2Agent
            self.base_agents['td_mpc2'] = TDMPC2Agent(
                self.state_dim,
                self.action_dim,
                self.base_agents_config['td_mpc2'],
                self.device
            )
    
    def get_base_predictions(self, state: np.ndarray, deterministic: bool = False) -> Dict[str, np.ndarray]:
        """Get predictions from all base agents"""
        predictions = {}
        
        for name, agent in self.base_agents.items():
            pred = agent.select_action(state, deterministic)
            predictions[name] = pred
        
        return predictions
    
    def construct_meta_state(self, state: np.ndarray, base_predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """Construct meta-state by concatenating original state with base predictions"""
        # Ensure consistent ordering
        ordered_predictions = []
        for name in sorted(self.base_agents.keys()):
            ordered_predictions.append(base_predictions[name])
        
        # Flatten predictions if needed
        flat_predictions = np.concatenate([p.flatten() for p in ordered_predictions])
        
        # Concatenate with original state
        meta_state = np.concatenate([state, flat_predictions])
        
        return meta_state
    
    def select_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """
        Select action using hierarchical ensemble approach
        """
        # Get base predictions
        base_predictions = self.get_base_predictions(state, deterministic)
        
        # Construct meta-state
        meta_state = self.construct_meta_state(state, base_predictions)
        
        # Get meta-agent prediction
        meta_action = self.meta_agent.select_action(meta_state, deterministic)
        
        # Alternative: weighted combination using meta-network
        if self.config.get('use_weighted_combination', False):
            meta_state_tensor = self.to_tensor(meta_state).unsqueeze(0)
            with torch.no_grad():
                weights = self.meta_network(meta_state_tensor)
                weights = F.softmax(weights, dim=-1).cpu().numpy()[0]
            
            # Weighted average of base predictions
            weighted_action = np.zeros_like(meta_action)
            for i, name in enumerate(sorted(self.base_agents.keys())):
                weighted_action += weights[i] * base_predictions[name]
            
            # Blend meta-agent and weighted predictions
            blend_factor = self.config.get('blend_factor', 0.5)
            final_action = blend_factor * meta_action + (1 - blend_factor) * weighted_action
            
            return final_action
        
        return meta_action
    
    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Update ensemble agent"""
        losses = {}
        
        # Update base agents if enabled
        if self.train_base_agents:
            for name, agent in self.base_agents.items():
                agent_losses = agent.update(batch)
                for key, value in agent_losses.items():
                    losses[f'{name}_{key}'] = value
        
        # Construct meta-batch
        states = batch['states'].cpu().numpy()
        meta_states = []
        
        for i in range(len(states)):
            base_preds = self.get_base_predictions(states[i], deterministic=True)
            meta_state = self.construct_meta_state(states[i], base_preds)
            meta_states.append(meta_state)
        
        meta_states = torch.FloatTensor(np.array(meta_states)).to(self.device)

        # Rebuild next_meta_states just like we did for states
        next_np = batch['next_states'].cpu().numpy()
        next_meta = []
        for i in range(len(next_np)):
            bp = self.get_base_predictions(next_np[i], deterministic=True)
            next_meta.append(self.construct_meta_state(next_np[i], bp))
        next_meta = torch.FloatTensor(np.array(next_meta)).to(self.device)
        
        meta_batch = {
            'states': meta_states,
            'actions': batch['actions'],
            'rewards': batch['rewards'],
            'next_states': next_meta,
            'dones': batch['dones']
        }
        
        # Update meta-agent
        if self.train_meta_agent:
            meta_losses = self.meta_agent.update(meta_batch)
            for key, value in meta_losses.items():
                losses[f'meta_{key}'] = value
        
        # Update meta-network if using weighted combination
        if self.config.get('use_weighted_combination', False):
            # Compute meta-network loss (MSE between weighted prediction and true action)
            weights = self.meta_network(meta_states)
            weights = F.softmax(weights, dim=-1)
            
            # Get base predictions for batch
            base_preds_batch = []
            for name in sorted(self.base_agents.keys()):
                agent_preds = []
                for i in range(len(states)):
                    pred = self.base_agents[name].select_action(states[i], deterministic=True)
                    agent_preds.append(pred)
                base_preds_batch.append(torch.FloatTensor(agent_preds).to(self.device))
            
            base_preds_batch = torch.stack(base_preds_batch, dim=1)
            
            # Weighted prediction
            weighted_pred = (weights.unsqueeze(-1) * base_preds_batch).sum(dim=1)
            
            # Meta-network loss
            meta_net_loss = F.mse_loss(weighted_pred, batch['actions'])
            
            self.meta_optimizer.zero_grad()
            meta_net_loss.backward()
            self.meta_optimizer.step()
            
            losses['meta_network_loss'] = meta_net_loss.item()
        
        self.training_steps += 1
        
        return losses
    
    def save(self, path: str):
        """Save ensemble agent"""
        import os
        os.makedirs(path, exist_ok=True)
        
        # Save base agents
        for name, agent in self.base_agents.items():
            agent.save(os.path.join(path, f'{name}_agent.pth'))
        
        # Save meta-agent
        self.meta_agent.save(os.path.join(path, 'meta_agent.pth'))
        
        # Save meta-network
        torch.save({
            'meta_network': self.meta_network.state_dict(),
            'meta_optimizer': self.meta_optimizer.state_dict(),
            'config': self.config,
            'training_steps': self.training_steps
        }, os.path.join(path, 'ensemble_meta.pth'))
    
    def load(self, path: str):
        """Load ensemble agent"""
        import os
        
        # Load base agents
        for name, agent in self.base_agents.items():
            agent_path = os.path.join(path, f'{name}_agent.pth')
            if os.path.exists(agent_path):
                agent.load(agent_path)
        
        # Load meta-agent
        meta_agent_path = os.path.join(path, 'meta_agent.pth')
        if os.path.exists(meta_agent_path):
            self.meta_agent.load(meta_agent_path)
        
        # Load meta-network
        meta_path = os.path.join(path, 'ensemble_meta.pth')
        if os.path.exists(meta_path):
            checkpoint = torch.load(meta_path, map_location=self.device)
            self.meta_network.load_state_dict(checkpoint['meta_network'])
            self.meta_optimizer.load_state_dict(checkpoint['meta_optimizer'])
            self.training_steps = checkpoint['training_steps']
