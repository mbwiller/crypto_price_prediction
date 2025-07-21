import pandas as pd
import numpy as np
import gym
from gym import spaces
from collections import deque

def add_technical_indicators(df, price_col='price'):
    """
    Add common technical indicators to DataFrame in-place.
    Requires a 'price' column; computes:
      - SMA (simple moving average) over 5 and 10 periods
      - EMA (exponential moving average) over 10 and 20 periods
      - RSI (relative strength index) with window=14
      - Bollinger Bands (20-period SMA ± 2*std)
      - Momentum (difference over 1 period)
    """
    # Simple Moving Averages
    df['SMA_5']  = df[price_col].rolling(window=5).mean()
    df['SMA_10'] = df[price_col].rolling(window=10).mean()
    
    # Exponential Moving Averages
    df['EMA_10'] = df[price_col].ewm(span=10, adjust=False).mean()
    df['EMA_20'] = df[price_col].ewm(span=20, adjust=False).mean()
    
    # Momentum
    df['MOM_1'] = df[price_col] - df[price_col].shift(1)
    
    # RSI
    delta = df[price_col].diff()
    up, down = delta.clip(lower=0), -delta.clip(upper=0)
    roll_up   = up.rolling(window=14).mean()
    roll_down = down.rolling(window=14).mean()
    rs = roll_up / roll_down
    df['RSI_14'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    rolling_mean = df[price_col].rolling(window=20).mean()
    rolling_std  = df[price_col].rolling(window=20).std()
    df['BB_upper'] = rolling_mean + 2 * rolling_std
    df['BB_lower'] = rolling_mean - 2 * rolling_std
    
    # After computing, you may drop initial NaNs
    df.dropna(inplace=True)
    return df

class CryptoTradingEnv(gym.Env):
    """
    State s_t:
      - feature vector at time t: market features (bid_qty, ask_qty, X1...X780, etc.)
      - current inventory/position held
      - previous action (optional)
      
    Action a_t:
      - Continuous in [-max_position, +max_position], indicating desired new position
    
    Reward r_t:
      - Profit & Loss: position_t * price_change_t
      - Transaction cost: -cost_coeff * |position_t - position_{t-1}|
      - Risk penalty: -risk_coeff * (position_t)^2
      - Frequency penalty: -freq_coeff * |position_t - position_{t-1}|
      - (Placeholder for CVaR-based penalty)
    """
    
    def __init__(self, df, max_position=1.0, cost_coeff=0.001, risk_coeff=0.01, freq_coeff=0.001, cvar_coeff=0.05, cvar_window=100, cvar_alpha=0.95):
        super().__init__()
        # DataFrame with datetime index and columns including 'label' = next-minute return
        self.df = df.reset_index(drop=True)
        self.features = df.columns.drop('label')

        self.max_position = max_position
        self.cost_coeff = cost_coeff
        self.risk_coeff = risk_coeff
        self.freq_coeff = freq_coeff
        
        self.cvar_coeff  = cvar_coeff
        self.cvar_window = cvar_window
        self.cvar_alpha  = cvar_alpha
        self.pnl_history = deque(maxlen=cvar_window)
        
        # Action space: continuous position in [-max_position, +max_position]
        self.action_space = spaces.Box(
            low=-self.max_position,
            high=+self.max_position,
            shape=(1,),
            dtype=np.float32
        )
        
        # Observation space: feature vector + inventory
        obs_dim = len(self.features) + 1  # +1 for current position
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        
        self.reset()
    
    def step(self, action):
        new_position = np.clip(action, -self.max_position, self.max_position)[0]
        price_change = self.df.loc[self.t, 'label']
        pnl = self.position * price_change
        
        # Update PnL history
        self.pnl_history.append(pnl)
        
        # Transaction & risk & frequency penalties
        transaction_cost = self.cost_coeff * abs(new_position - self.prev_position)
        risk_penalty     = self.risk_coeff * (new_position ** 2)
        freq_penalty     = self.freq_coeff * abs(new_position - self.prev_position)
        
        # Compute CVaR penalty if enough history
        cvar_penalty = 0.0
        if len(self.pnl_history) >= self.cvar_window:
            # Compute VaR at (1 - alpha) quantile of losses
            losses = np.array([-x for x in self.pnl_history])
            var_level = np.quantile(losses, self.cvar_alpha)
            # CVaR = average of losses ≥ VaR
            tail_losses = losses[losses >= var_level]
            cvar = tail_losses.mean() if len(tail_losses) > 0 else 0.0
            cvar_penalty = self.cvar_coeff * cvar
        
        # Total reward
        reward = pnl - transaction_cost - risk_penalty - freq_penalty - cvar_penalty
        
        # Advance state
        self.prev_position = new_position
        self.position      = new_position
        self.t += 1
        done = self.t >= len(self.df)
        obs  = self._get_obs()
        info = {
            'pnl': pnl,
            'transaction_cost': transaction_cost,
            'risk_penalty': risk_penalty,
            'freq_penalty': freq_penalty,
            'cvar_penalty': cvar_penalty
        }
        return obs, reward, done, info
    
    def _get_obs(self):
        # Extract market features at time t
        feature_vals = self.df.loc[self.t, self.features].values.astype(np.float32)
        # Append current position
        obs = np.concatenate([feature_vals, [self.position]], axis=0)
        return obs
    
    def reset(self):
        # Start at timestep 0, with zero position
        self.t = 0
        self.position = 0.0
        # Previous position (for transaction cost / frequency penalty)
        self.prev_position = 0.0
        
        return self._get_obs()
    
    def render(self, mode='human'):
        print(f"Step: {self.t}, Position: {self.position:.4f}")

# === Usage Example ===
if __name__ == "__main__":
    # Load train data
    df_train = pd.read_parquet("train.parquet", engine="pyarrow")
    
    # Instantiate environment
    env = CryptoTradingEnv(df_train)
    
    # Sample random steps
    obs = env.reset()
    for _ in range(5):
        action = env.action_space.sample()      # random action
        obs, reward, done, info = env.step(action)
        print(f"Obs shape: {obs.shape}, Reward: {reward:.4f}, Info: {info}")
        if done:
            break