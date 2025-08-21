import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
from typing import Dict, Any
from .base_agent import BaseAgent

class Encoder(nn.Module):
    """Encodes raw state s -> latent z"""
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, latent_dim)
        )
    def forward(self, s):
        return self.net(s)

class DynamicsModel(nn.Module):
    """Predicts next latent z' given current z and action a"""
    def __init__(self, latent_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + action_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, latent_dim)
        )
    def forward(self, z, a):
        x = torch.cat([z, a], dim=-1)
        return self.net(x)

class RewardModel(nn.Module):
    """Predicts reward given latent z and action a"""
    def __init__(self, latent_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + action_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1)
        )
    def forward(self, z, a):
        x = torch.cat([z, a], dim=-1)
        return self.net(x).squeeze(-1)

class QNetwork(nn.Module):
    """Estimates Q(z,a)"""
    def __init__(self, latent_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + action_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1)
        )
    def forward(self, z, a):
        x = torch.cat([z, a], dim=-1)
        return self.net(x).squeeze(-1)

class Policy(nn.Module):
    """Tanh-Gaussian policy: outputs mean and log_std for action distribution"""
    def __init__(self, latent_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU()
        )
        self.mean_layer = nn.Linear(256, action_dim)
        self.log_std_layer = nn.Linear(256, action_dim)
    def forward(self, z):
        h = self.net(z)
        mu = self.mean_layer(h)
        log_std = self.log_std_layer(h).clamp(-5, 2)
        std = torch.exp(log_std)
        return mu, std
    def sample(self, z):
        mu, std = self(z)
        dist = torch.distributions.Normal(mu, std)
        x = dist.rsample()
        a = torch.tanh(x)
        logp = dist.log_prob(x) - torch.log(1 - a.pow(2) + 1e-6)
        logp = logp.sum(-1)
        return a, logp

class ReplayBuffer:
    """Simple FIFO experience replay"""
    def __init__(self, max_size=1000000):
        self.buffer = deque(maxlen=max_size)
    def add(self, transition):  # (s,a,mu,r,s')
        self.buffer.append(transition)
    def sample(self, batch_size):
        idx = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in idx]
        return map(np.stack, zip(*batch))

class TDMPC2Agent(BaseAgent):
    """TD-M(PC)^2 agent: model-based RL with policy constraint"""
    def __init__(self, state_dim: int, action_dim: int, config: Dict[str, Any], device: str = 'cpu'):
        super().__init__(state_dim, action_dim, config, device)
        # Hyperparameters from config
        latent_dim = config.get('latent_dim', 32)
        self.horizon = config.get('horizon', 5)
        # Device
        self.device = self.device  # inherited from BaseAgent
        # Replay buffer
        self.buffer = deque(maxlen=config.get('buffer_size', 1_000_000))

        # Build networks using state_dim / action_dim
        self.encoder = Encoder(state_dim, latent_dim).to(self.device)
        self.dynamics = DynamicsModel(latent_dim, action_dim).to(self.device)
        self.reward_model = RewardModel(latent_dim, action_dim).to(self.device)
        self.q1 = QNetwork(latent_dim, action_dim).to(self.device)
        self.q2 = QNetwork(latent_dim, action_dim).to(self.device)
        self.q1_target = QNetwork(latent_dim, action_dim).to(self.device)
        self.q2_target = QNetwork(latent_dim, action_dim).to(self.device)
        self.policy = Policy(latent_dim, action_dim).to(self.device)
        # Copy targets
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        # Optimizers
        lr_model = config.get('lr_model', 1e-3)
        lr_value = config.get('lr_value', 1e-3)
        lr_policy = config.get('lr_policy', 1e-4)
        self.model_optimizer = optim.Adam(
            list(self.encoder.parameters()) +
            list(self.dynamics.parameters()) +
            list(self.reward_model.parameters()),
            lr=lr_model
        )
        self.value_optimizer = optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()),
            lr=lr_value
        )
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr_policy)

        # Other hyperparameters
        self.gamma = config.get('gamma', 0.99)
        self.rho = config.get('rho', 0.995)
        self.alpha = config.get('alpha', 0.1)
        self.beta = config.get('beta', 0.1)
        self.horizon = config.get('horizon', 5)

    def plan_action(self, s):
        """MPPI planning in latent space: returns first action a_t"""
        # encode
        z = self.encoder(torch.from_numpy(s).float().to(self.device))
        # sample action sequences, evaluate returns, fit mu, sigma (omitted for brevity)
        # placeholder: use nominal policy directly
        with torch.no_grad():
            a, _ = self.policy.sample(z)
        return a.cpu().numpy()

    def train(self, total_steps=100000, batch_size=256, collect_freq=1):
        obs = self.env.reset()
        for step in range(total_steps):
            # ---------- collect data ----------
            a = self.plan_action(obs)
            mu = a.copy()  # planner's mean policy
            next_obs, r, done, _ = self.env.step(a)
            # store (s, a, mu, r, s')
            self.buffer.add((obs, a, mu, r, next_obs))
            obs = next_obs if not done else self.env.reset()
            # ---------- update ---- every collect_freq steps ----------
            if step % collect_freq == 0 and len(self.buffer.buffer) >= batch_size:
                self.update(batch_size)
        return

    def update(self, batch_size):
        """One gradient step: model, Q, policy"""
        s, a, mu, r, s2 = self.buffer.sample(batch_size)
        # to tensors
        s = torch.from_numpy(s).float().to(self.device)
        a = torch.from_numpy(a).float().to(self.device)
        mu = torch.from_numpy(mu).float().to(self.device)
        r = torch.from_numpy(r).float().to(self.device)
        s2 = torch.from_numpy(s2).float().to(self.device)
        # encode
        z = self.encoder(s)
        z2 = self.encoder(s2).detach()
        # model losses (eq 3)
        z2_pred = self.dynamics(z, a)
        r_pred = self.reward_model(z, a)
        # target Q for model loss
        with torch.no_grad():
            q1_next = self.q1_target(z2, self.policy.sample(z2)[0])
            q2_next = self.q2_target(z2, self.policy.sample(z2)[0])
            q_target = torch.min(q1_next, q2_next)
        # Q bootstrap
        q_pred = torch.min(self.q1(z, a), self.q2(z, a))
        # model + reward + Q losses
        model_loss = nn.MSELoss()(z2_pred, z2) + nn.MSELoss()(r_pred, r) + nn.MSELoss()(q_pred, q_target)
        self.model_optimizer.zero_grad(); model_loss.backward(); self.model_optimizer.step()
        # Q-function update (TD) (eq 9)
        with torch.no_grad():
            a2, logp2 = self.policy.sample(z2)
            q_backup = r + self.gamma * (torch.min(
                self.q1_target(z2, a2),
                self.q2_target(z2, a2)
            ) - self.alpha * logp2)
        q1_loss = nn.MSELoss()(self.q1(z, a), q_backup)
        q2_loss = nn.MSELoss()(self.q2(z, a), q_backup)
        self.value_optimizer.zero_grad(); (q1_loss + q2_loss).backward(); self.value_optimizer.step()
        # Constrained policy update (eq 10)
        a_pi, logp_pi = self.policy.sample(z)
        q1_pi = self.q1(z, a_pi)
        q2_pi = self.q2(z, a_pi)
        q_pi = torch.min(q1_pi, q2_pi)
        # policy loss: maximize Q - alpha*entropy + beta*log mu
        log_mu = torch.log(torch.clamp(mu, 1e-6, 1.0))
        policy_loss = -(q_pi - self.alpha * logp_pi + self.beta * log_mu).mean()
        self.policy_optimizer.zero_grad(); policy_loss.backward(); self.policy_optimizer.step()
        # Polyak update targets
        for p, pt in zip(self.q1.parameters(), self.q1_target.parameters()):
            pt.data.mul_(self.rho); pt.data.add_((1-self.rho)*p.data)
        for p, pt in zip(self.q2.parameters(), self.q2_target.parameters()):
            pt.data.mul_(self.rho); pt.data.add_((1-self.rho)*p.data)

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save({
            'enc': self.encoder.state_dict(),
            'dyn': self.dynamics.state_dict(),
            'rew': self.reward_model.state_dict(),
            'q1': self.q1.state_dict(),
            'q2': self.q2.state_dict(),
            'pi': self.policy.state_dict()
        }, os.path.join(path, 'tdmpc2.pth'))

    def load(self, path):
        data = torch.load(os.path.join(path, 'tdmpc2.pth'), map_location=self.device)
        self.encoder.load_state_dict(data['enc'])
        self.dynamics.load_state_dict(data['dyn'])
        self.reward_model.load_state_dict(data['rew'])
        self.q1.load_state_dict(data['q1'])
        self.q2.load_state_dict(data['q2'])
        self.policy.load_state_dict(data['pi'])

    def predict(self, s):
        """Return action given observation s (for meta-env usage)"""
        a = self.plan_action(s)
        return np.array([a]), None

    # --- Adapter to BaseAgent API ---
    def select_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Conform to BaseAgent: wrap predict()"""
        action, _ = self.predict(state)
        return action

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """No-op or custom training — we typically freeze this expert."""
        return {}
