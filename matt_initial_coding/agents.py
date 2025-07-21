import os
import numpy as np
import pandas as pd
import gym
from gym import spaces

# Stable-Baselines3 algorithms
from stable_baselines3 import SAC, PPO, DDPG

# Import environment and any custom agent implementations
from env import CryptoTradingEnv
from TD_MPC2_agent import TD_MPC2Agent
from env import add_technical_indicators

df = pd.read_parquet("train.parquet", engine="pyarrow")
df = add_technical_indicators(df, price_col="price")
# =============================================================================
# 1) Train Level-0 Base Agents
# =============================================================================

def train_base_agents(df, model_dir="models/base", timesteps=200_000):
    """
    Trains four base agents on the same environment and returns them.
    """
    os.makedirs(model_dir, exist_ok=True)
    
    env = CryptoTradingEnv(df)
    
    # -- SAC --
    sac = SAC("MlpPolicy", env, verbose=1)
    sac.learn(total_timesteps=timesteps)
    sac.save(f"{model_dir}/sac_base")
    
    # -- PPO --
    ppo = PPO("MlpPolicy", env, verbose=1)
    ppo.learn(total_timesteps=timesteps)
    ppo.save(f"{model_dir}/ppo_base")
    
    # -- DDPG --
    ddpg = DDPG("MlpPolicy", env, verbose=1)
    ddpg.learn(total_timesteps=timesteps)
    ddpg.save(f"{model_dir}/ddpg_base")
    
    # -- TD-M(PC)^2 (Model-Based placeholder) --
    tdmpc = TD_MPC2Agent(env)            # you must implement this class
    tdmpc.train(total_steps=timesteps)  # adjust to your API
    tdmpc.save(f"{model_dir}/tdmpc2_base")
    
    return sac, ppo, ddpg, tdmpc

# =============================================================================
# 2) Meta-Environment Wrapper
# =============================================================================

class MetaTradingEnv(gym.Env):
    """
    Wraps the underlying CryptoTradingEnv and N base agents into a single
    meta-environment. At each step:
      1. Observe raw state s_t from base env
      2. Query each base_agent.predict(s_t) -> continuous action_i
      3. Form meta_state = [s_t, action_1, ..., action_N]
      4. Execute meta_action in base env, returning (next_meta_state, reward, done, info)
    """
    def __init__(self, base_env, base_agents):
        super().__init__()
        self.base_env    = base_env
        self.base_agents = base_agents  # list of trained agents
        
        # Original obs_dim + one action per base agent
        orig_dim = self.base_env.observation_space.shape[0]
        N = len(base_agents)
        meta_dim = orig_dim + N
        
        # Meta observation: continuous unbounded
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(meta_dim,), dtype=np.float32
        )
        # Meta action: same as base action (continuous position)
        self.action_space = self.base_env.action_space
    
    def reset(self):
        s = self.base_env.reset()
        return self._make_meta_obs(s)
    
    def _make_meta_obs(self, s):
        """
        Given raw state s, query each base agent for its action
        and concatenate into a meta-state vector.
        """
        actions = []
        for agent in self.base_agents:
            # For SB3 agents: .predict returns (action, _)
            a, _ = agent.predict(s, deterministic=True)
            # ensure it’s a scalar
            actions.append(float(np.asarray(a).reshape(-1)[0]))
        
        # Concatenate raw state + base actions
        return np.concatenate([s, np.array(actions, dtype=np.float32)], axis=0)
    
    def step(self, meta_action):
        # Execute the chosen meta_action in the base env
        s_next, reward, done, info = self.base_env.step(meta_action)
        # Build next meta-state
        meta_obs = self._make_meta_obs(s_next)
        return meta_obs, reward, done, info
    
    def render(self, mode="human"):
        self.base_env.render(mode=mode)

# =============================================================================
# 3) Train the "Alpha" (Meta) Agent
# =============================================================================

def train_meta_agent(df, base_agents, model_dir="models/meta", timesteps=200_000):
    os.makedirs(model_dir, exist_ok=True)
    
    # Instantiate a fresh copy of the base environment
    base_env = CryptoTradingEnv(df)
    # Wrap it with our MetaTradingEnv
    meta_env = MetaTradingEnv(base_env, base_agents)
    
    # Use SAC for the alpha agent—off‐policy, continuous action, entropy bonus
    alpha = SAC("MlpPolicy", meta_env, verbose=1)
    alpha.learn(total_timesteps=timesteps)
    alpha.save(f"{model_dir}/alpha_agent")
    
    return alpha

# =============================================================================
# 4) Main script
# =============================================================================

if __name__ == "__main__":
    # 1) Load training data
    df_train = pd.read_parquet("train.parquet", engine="pyarrow")
    
    # 2) (Optionally) add any technical indicators / pre‐processing here
    # df_train = add_technical_indicators(df_train, price_col='price')
    
    # 3) Train Level-0 base agents
    sac, ppo, ddpg, tdmpc = train_base_agents(df_train, timesteps=100_000)
    
    # 4) Train the Level-1 alpha agent
    alpha = train_meta_agent(
        df_train,
        base_agents=[sac, ppo, ddpg, tdmpc],
        timesteps=100_000
    )
    
    print("All models trained and saved under 'models/' directory.")