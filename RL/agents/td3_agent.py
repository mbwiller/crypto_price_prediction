import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Dict, Any, List

from .base_agent import BaseAgent
from ..networks.actor_networks import DeterministicActor
from ..networks.critic_networks import TwinQNetwork


class TD3Agent(BaseAgent):
    """Twin Delayed Deep Deterministic Policy Gradient agent."""

    def __init__(self, state_dim: int, action_dim: int, config: Dict[str, Any], device: str = "cuda"):
        super().__init__(state_dim, action_dim, config, device)

        self.gamma = config.get("gamma", 0.99)
        self.tau = config.get("tau", 0.005)
        self.actor_lr = config.get("actor_lr", 1e-4)
        self.critic_lr = config.get("critic_lr", 1e-3)
        self.hidden_dims: List[int] = config.get("hidden_dims", [256, 256])
        self.policy_delay = config.get("policy_delay", 2)
        self.target_noise_std = config.get("target_noise_std", 0.2)
        self.target_noise_clip = config.get("target_noise_clip", 0.5)

        self.actor = DeterministicActor(state_dim, action_dim, self.hidden_dims).to(self.device)
        self.actor_target = DeterministicActor(state_dim, action_dim, self.hidden_dims).to(self.device)
        self.critic = TwinQNetwork(state_dim, action_dim, self.hidden_dims).to(self.device)
        self.critic_target = TwinQNetwork(state_dim, action_dim, self.hidden_dims).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.critic_lr)

    def select_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        state_tensor = self.to_tensor(state).unsqueeze(0)
        with torch.no_grad():
            action = self.actor(state_tensor)
        return action.cpu().numpy()[0]

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        states = batch["states"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        next_states = batch["next_states"]
        dones = batch["dones"]

        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            noise = torch.normal(0, self.target_noise_std, size=next_actions.shape, device=self.device)
            noise = noise.clamp(-self.target_noise_clip, self.target_noise_clip)
            next_actions = (next_actions + noise).clamp(-1.0, 1.0)
            q1_next, q2_next = self.critic_target(next_states, next_actions)
            q_target = rewards + self.gamma * (1 - dones) * torch.min(q1_next, q2_next)

        q1, q2 = self.critic(states, actions)
        critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()

        actor_loss = torch.tensor(0.0)
        if self.training_steps % self.policy_delay == 0:
            actor_loss = -self.critic.q1(states, self.actor(states)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            self.actor_optimizer.step()
            self.soft_update(self.actor_target, self.actor, self.tau)

        self.soft_update(self.critic_target, self.critic, self.tau)
        self.training_steps += 1

        return {"critic_loss": float(critic_loss.item()), "actor_loss": float(actor_loss.item())}

    def save(self, path: str):
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "actor_target": self.actor_target.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "training_steps": self.training_steps,
            "config": self.config,
        }, path)

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.actor_target.load_state_dict(checkpoint["actor_target"])
        self.critic_target.load_state_dict(checkpoint["critic_target"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        self.training_steps = checkpoint.get("training_steps", 0)
