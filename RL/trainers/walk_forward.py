import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import logging
from tqdm import tqdm
import torch
from collections import deque

from ..agents.base_agent import BaseAgent
from ..replay_buffers.replay_buffer import ReplayBuffer
from ..utils.metrics import calculate_metrics
from ..utils.data_loader import CryptoDataLoader

class WalkForwardTrainer:
    """
    Walk-forward validation trainer for crypto price prediction
    Simulates real-world trading conditions
    """
    
    def __init__(self,
                 agent: BaseAgent,
                 mdp,
                 config: Dict[str, Any]):
        
        self.agent = agent
        self.mdp = mdp
        self.config = config
        
        # Walk-forward parameters
        self.initial_train_window = config.get('initial_train_window', 2 * 365 * 24 * 60)  # 2 years
        self.validation_window = config.get('validation_window', 180 * 24 * 60)  # 6 months
        self.step_size = config.get('step_size', 30 * 24 * 60)  # 1 month
        self.retrain_frequency = config.get('retrain_frequency', 'monthly')
        
        # Training parameters
        self.batch_size = config.get('batch_size', 256)
        self.replay_buffer_size = config.get('replay_buffer_size', 1000000)
        self.learning_starts = config.get('learning_starts', 10000)
        self.update_frequency = config.get('update_frequency', 1)
        self.gradient_steps = config.get('gradient_steps', 1)
        
        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(
            self.replay_buffer_size,
            self.mdp.state_dim if hasattr(self.mdp, 'state_dim') else None
        )
        
        # Logging
        self.logger = logging.getLogger(__name__)
        
        # Metrics tracking
        self.train_metrics = []
        self.val_metrics = []
        self.predictions = []
        
    def train(self, data: pd.DataFrame, num_epochs: int = 1) -> Dict[str, List[float]]:
        """
        Train agent using walk-forward validation
        """
        self.logger.info(f"Starting walk-forward training with {len(data)} samples")
        
        # Initialize windows
        train_start = 0
        train_end = self.initial_train_window
        val_start = train_end
        val_end = val_start + self.validation_window
        
        epoch_results = []
        
        while val_end <= len(data):
            self.logger.info(f"Training window: {train_start} to {train_end}")
            self.logger.info(f"Validation window: {val_start} to {val_end}")
            
            # Get data windows
            train_data = data.iloc[train_start:train_end]
            val_data = data.iloc[val_start:val_end]
            
            # Train on current window
            for epoch in range(num_epochs):
                train_metrics = self._train_epoch(train_data, epoch)
                self.train_metrics.append(train_metrics)
            
            # Validate
            val_metrics, predictions = self._validate(val_data)
            self.val_metrics.append(val_metrics)
            self.predictions.extend(predictions)
            
            epoch_results.append({
                'train_window': (train_start, train_end),
                'val_window': (val_start, val_end),
                'train_metrics': train_metrics,
                'val_metrics': val_metrics
            })
            
            # Move windows forward
            if self.retrain_frequency == 'expanding':
                # Expanding window
                train_end += self.step_size
            else:
                # Rolling window
                train_start += self.step_size
                train_end += self.step_size
            
            val_start += self.step_size
            val_end += self.step_size
        
        return {
            'epoch_results': epoch_results,
            'final_metrics': self._calculate_final_metrics()
        }
    
    def _train_epoch(self, data: pd.DataFrame, epoch: int) -> Dict[str, float]:
        """Train for one epoch on given data"""
        
        # Reset MDP if needed
        if hasattr(self.mdp, 'reset'):
            state = self.mdp.reset(data.iloc[:100])  # Use first 100 rows for initialization
        
        total_reward = 0
        episode_rewards = []
        losses = []
        
        # Progress bar
        pbar = tqdm(range(100, len(data)), desc=f"Epoch {epoch}")
        
        for idx in pbar:
            # Get current row
            current_row = data.iloc[idx]
            
            # Select action
            action = self.agent.select_action(state, deterministic=False)
            
            # Step environment
            next_state, reward, done, info = self.mdp.step(action, current_row)
            
            # Store transition
            self.replay_buffer.add(state, action, reward, next_state, done)
            
            # Update agent
            if len(self.replay_buffer) >= self.learning_starts and idx % self.update_frequency == 0:
                for _ in range(self.gradient_steps):
                    batch = self.replay_buffer.sample(self.batch_size)
                    loss_dict = self.agent.update(batch)
                    losses.append(loss_dict)
            
            # Track metrics
            total_reward += reward
            if done:
                episode_rewards.append(total_reward)
                total_reward = 0
                if hasattr(self.mdp, 'reset'):
                    state = self.mdp.reset(data.iloc[max(0, idx-100):idx])
            else:
                state = next_state
            
            # Update progress bar
            if len(losses) > 0:
                avg_loss = np.mean([l.get('critic_loss', 0) for l in losses[-100:]])
                pbar.set_postfix({'reward': f'{reward:.4f}', 'avg_loss': f'{avg_loss:.4f}'})
        
        # Calculate epoch metrics
        metrics = {
            'mean_reward': np.mean(episode_rewards) if episode_rewards else total_reward / len(data),
            'std_reward': np.std(episode_rewards) if episode_rewards else 0,
            'mean_loss': np.mean([l.get('critic_loss', 0) for l in losses]) if losses else 0,
            'num_episodes': len(episode_rewards)
        }
        
        return metrics
    
    def _validate(self, data: pd.DataFrame) -> Tuple[Dict[str, float], List[Dict]]:
        """Validate on given data"""
        
        predictions = []
        actual_returns = []
        prediction_errors = []
        
        # Reset MDP
        if hasattr(self.mdp, 'reset'):
            state = self.mdp.reset(data.iloc[:100])
        
        # Disable gradient computation
        with torch.no_grad():
            for idx in tqdm(range(100, len(data)), desc="Validating"):
                current_row = data.iloc[idx]
                
                # Get deterministic action (prediction)
                action = self.agent.select_action(state, deterministic=True)
                
                # Step environment
                next_state, reward, done, info = self.mdp.step(action, current_row)
                
                # Store predictions
                predictions.append({
                    'index': idx,
                    'predicted_return': float(action),
                    'actual_return': info['actual_return'],
                    'prediction_error': info['prediction_error']
                })
                
                actual_returns.append(info['actual_return'])
                prediction_errors.append(info['prediction_error'])
                
                # Update state
                if done and hasattr(self.mdp, 'reset'):
                    state = self.mdp.reset(data.iloc[max(0, idx-100):idx])
                else:
                    state = next_state
        
        # Calculate validation metrics
        metrics = calculate_metrics(
            np.array([p['predicted_return'] for p in predictions]),
            np.array(actual_returns)
        )
        
        return metrics, predictions
    
    def _calculate_final_metrics(self) -> Dict[str, float]:
        """Calculate final aggregated metrics"""
        
        # Aggregate validation metrics
        all_val_metrics = {}
        for key in self.val_metrics[0].keys():
            values = [m[key] for m in self.val_metrics]
            all_val_metrics[f'mean_{key}'] = np.mean(values)
            all_val_metrics[f'std_{key}'] = np.std(values)
        
        # Calculate prediction accuracy
        all_predictions = []
        all_actuals = []
        for pred in self.predictions:
            all_predictions.append(pred['predicted_return'])
            all_actuals.append(pred['actual_return'])
        
        final_metrics = calculate_metrics(
            np.array(all_predictions),
            np.array(all_actuals)
        )
        
        # Add walk-forward specific metrics
        final_metrics.update(all_val_metrics)
        final_metrics['total_predictions'] = len(self.predictions)
        
        return final_metrics
    
    def save_predictions(self, path: str):
        """Save predictions to file"""
        df = pd.DataFrame(self.predictions)
        df.to_csv(path, index=False)
        self.logger.info(f"Saved {len(df)} predictions to {path}")
