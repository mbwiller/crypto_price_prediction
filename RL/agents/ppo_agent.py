import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Dict, Any, List

from .base_agent import BaseAgent
from ..networks.actor_networks import SquashedGaussianActor
from ..networks.critic_networks import ValueNetwork


class PPOAgent(BaseAgent):
    """Proximal Policy Optimization agent for continuous actions."""

    def __init__(self, state_dim: int, action_dim: int, config: Dict[str, Any], device: str = "cuda"):
        super().__init__(state_dim, action_dim, config, device)

        self.gamma = config.get("gamma", 0.99)
        self.lam = config.get("lambda", 0.95)
        self.clip_eps = config.get("clip_eps", 0.2)
        self.entropy_coef = config.get("entropy_coef", 0.0)
        self.learning_rate = config.get("learning_rate", 3e-4)
        self.hidden_dims: List[int] = config.get("hidden_dims", [256, 256])
        self.train_iters = config.get("train_iters", 10)

        self.actor = SquashedGaussianActor(state_dim, action_dim, self.hidden_dims).to(self.device)
        self.critic = ValueNetwork(state_dim, self.hidden_dims).to(self.device)

        self.optimizer = optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()), lr=self.learning_rate)

    def _evaluate_actions(self, states: torch.Tensor, actions: torch.Tensor) -> Dict[str, torch.Tensor]:
        mean, log_std = self.actor.forward(states)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        # invert tanh
        atanh_actions = torch.atanh(torch.clamp(actions, -0.999, 0.999))
        log_probs = normal.log_prob(atanh_actions) - torch.log(1 - actions.pow(2) + 1e-6)
        log_probs = log_probs.sum(-1, keepdim=True)
        entropy = normal.entropy().sum(-1, keepdim=True)
        return {"log_probs": log_probs, "entropy": entropy}

    def select_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        state_tensor = self.to_tensor(state).unsqueeze(0)
        with torch.no_grad():
            action, _, mean_action = self.actor.sample(state_tensor)
            action = mean_action if deterministic else action
        return action.cpu().numpy()[0]

    def _compute_gae(self, rewards, dones, values, next_values):
        deltas = rewards + self.gamma * (1 - dones) * next_values - values
        adv = torch.zeros_like(rewards)
        gae = 0
        for t in reversed(range(len(rewards))):
            gae = deltas[t] + self.gamma * self.lam * (1 - dones[t]) * gae
            adv[t] = gae
        returns = adv + values
        return adv, returns

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        states = batch["states"]
        actions = batch["actions"]
        rewards = batch["rewards"].squeeze(-1)
        next_states = batch["next_states"]
        dones = batch["dones"].squeeze(-1)

        with torch.no_grad():
            values = self.critic(states).squeeze(-1)
            next_values = self.critic(next_states).squeeze(-1)
            advantages, returns = self._compute_gae(rewards, dones, values, next_values)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            old_log_probs = self._evaluate_actions(states, actions)["log_probs"].detach()

        losses = {}
        for _ in range(self.train_iters):
            eval_out = self._evaluate_actions(states, actions)
            log_probs = eval_out["log_probs"]
            entropy = eval_out["entropy"]
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(self.critic(states).squeeze(-1), returns)
            entropy_loss = -entropy.mean()

            loss = policy_loss + 0.5 * value_loss + self.entropy_coef * entropy_loss

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(list(self.actor.parameters()) + list(self.critic.parameters()), 1.0)
            self.optimizer.step()

        self.training_steps += 1
        losses["policy_loss"] = float(policy_loss.item())
        losses["value_loss"] = float(value_loss.item())
        losses["entropy"] = float(entropy.mean().item())
        return losses

    def save(self, path: str):
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "training_steps": self.training_steps,
            "config": self.config,
        }, path)

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.training_steps = checkpoint.get("training_steps", 0)
