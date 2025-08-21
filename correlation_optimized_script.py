"""
Pearson Correlation Optimized Crypto RL Training Script
Specifically optimized for Pearson correlation coefficient evaluation metric
"""

import os
import sys
import pandas as pd
import numpy as np
import torch
import logging
import yaml
from datetime import datetime
from typing import Dict, List, Any
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

# Import ALL implemented components
from crypto_MDP.mdp import CryptoPricePredictionMDP
from crypto_MDP.indicators import TechnicalIndicators
from crypto_MDP.regime import VolatilityRegimeDetector, MarketRegimeClassifier
from crypto_MDP.microstructure import MarketMicrostructure
from crypto_MDP.feature_selection import FeatureSelector

from RL.agents.ensemble_agent import EnsembleAgent
from RL.utils.data_loader import CryptoDataLoader
from RL.utils.metrics import calculate_metrics, calculate_trading_metrics
from RL.utils.config import load_config
from RL.replay_buffers.replay_buffer import ReplayBuffer

class CorrelationOptimizedMDP(CryptoPricePredictionMDP):
    """MDP optimized for Pearson correlation"""
    
    def _calculate_reward(self, action: float, actual_return: float, next_row: pd.Series) -> tuple:
        """Reward function optimized for Pearson correlation"""
        
        # Store predictions and actuals for correlation calculation
        if not hasattr(self, 'prediction_history'):
            self.prediction_history = []
            self.actual_history = []
        
        # Ensure action is a scalar
        if isinstance(action, np.ndarray):
            action = float(action.flatten()[0])
        else:
            action = float(action)
        
        self.prediction_history.append(action)
        self.actual_history.append(actual_return)
        
        # Keep only recent history (last 1000 samples)
        if len(self.prediction_history) > 1000:
            self.prediction_history = self.prediction_history[-1000:]
            self.actual_history = self.actual_history[-1000:]
        
        # Base reward: negative squared error (for accuracy)
        base_reward = -(action - actual_return) ** 2
        
        # Correlation bonus (calculated every 100 steps)
        correlation_bonus = 0.0
        current_correlation = 0.0
        if len(self.prediction_history) >= 100 and len(self.prediction_history) % 100 == 0:
            try:
                # Calculate rolling Pearson correlation with proper arrays
                pred_array = np.array(self.prediction_history, dtype=np.float64)
                actual_array = np.array(self.actual_history, dtype=np.float64)
                
                # Ensure 1D arrays
                pred_array = pred_array.flatten()
                actual_array = actual_array.flatten()
                
                corr = np.corrcoef(pred_array, actual_array)[0, 1]
                if not np.isnan(corr):
                    current_correlation = corr
                    # Reward high correlation, penalize negative correlation
                    correlation_bonus = corr * 10.0  # Scale factor
            except Exception as e:
                correlation_bonus = 0.0
                current_correlation = 0.0
        
        # Regularization: penalize extreme predictions
        magnitude_penalty = 0.0
        if abs(action) > 0.1:  # Penalize predictions > 10%
            magnitude_penalty = -abs(action) * 5.0
        
        # Final reward
        total_reward = base_reward + correlation_bonus + magnitude_penalty
        
        reward_components = {
            'base_reward': base_reward,
            'correlation_bonus': correlation_bonus,
            'magnitude_penalty': magnitude_penalty,
            'current_correlation': current_correlation,
            'prediction_error': abs(action - actual_return)
        }
        
        return total_reward, reward_components

def setup_logging(output_dir: str):
    """Setup production logging"""
    os.makedirs(output_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(output_dir, 'correlation_optimized_training.log'), encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def create_correlation_optimized_ensemble(state_dim: int, action_dim: int, device: str, logger: logging.Logger):
    """Create ensemble optimized for correlation"""
    
    logger.info("Creating correlation-optimized ensemble...")
    
    # Configuration optimized for correlation
    config = {
        'base_agents': {
            'sac': {
                'enabled': True,
                'gamma': 0.99,
                'tau': 0.01,  # Faster updates for correlation
                'alpha': 0.1,  # Lower entropy for more focused predictions
                'automatic_entropy_tuning': False,
                'hidden_dims': [256, 256],
                'learning_rate': 0.0001  # Lower LR for stability
            },
            'ddpg': {
                'enabled': True,
                'gamma': 0.99,
                'tau': 0.01,
                'hidden_dims': [256, 256],
                'learning_rate': 0.0001
            },
            'ppo': {
                'enabled': False,  # Keep disabled for stability
            },
            'td3': {
                'enabled': True,
                'gamma': 0.99,
                'tau': 0.01,
                'noise_std': 0.1,  # Lower noise for more consistent predictions
                'noise_clip': 0.3,
                'policy_delay': 2,
                'hidden_dims': [256, 256],
                'learning_rate': 0.0001
            },
            'td_mpc2': {
                'enabled': True,
                'gamma': 0.99,
                'tau': 0.01,
                'horizon': 3,  # Shorter horizon for faster training
                'hidden_dims': [256, 128],
                'latent_dim': 128,
                'learning_rate': 0.0001,
                'beta': 0.05  # Lower constraint
            }
        },
        'meta_agent_config': {
            'gamma': 0.99,
            'tau': 0.01,
            'alpha': 0.1,
            'automatic_entropy_tuning': False,
            'hidden_dims': [256, 256],
            'learning_rate': 0.00005
        },
        'meta_hidden_dims': [256, 128],
        'meta_learning_rate': 0.00005,
        'train_base_agents': True,
        'train_meta_agent': True,
        'use_weighted_combination': True,
        'blend_factor': 0.8  # Trust meta-agent more
    }
    
    ensemble_agent = EnsembleAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        config=config,
        device=device
    )
    
    logger.info(f"Correlation-optimized ensemble created with {len(ensemble_agent.base_agents)} agents")
    return ensemble_agent

def correlation_optimized_training(data_file: str, test_file: str, output_dir: str, logger: logging.Logger):
    """Full training pipeline optimized for Pearson correlation"""
    
    logger.info("="*80)
    logger.info("CORRELATION-OPTIMIZED TRAINING PIPELINE")
    logger.info("="*80)
    
    # Load data
    logger.info("Loading training and test datasets...")
    train_df = pd.read_parquet(data_file)
    test_df = pd.read_parquet(test_file)
    
    logger.info(f"Training data: {len(train_df):,} rows")
    logger.info(f"Test data: {len(test_df):,} rows")
    
    # Data preparation (same as before)
    for df, name in [(train_df, 'train'), (test_df, 'test')]:
        if 'close' not in df.columns:
            if name == 'train':
                base_price = 50000.0
                close_prices = np.zeros(len(df))
                close_prices[0] = base_price
                for i in range(1, len(df)):
                    return_val = df['label'].iloc[i-1] if 'label' in df.columns and not pd.isna(df['label'].iloc[i-1]) else 0
                    return_val = np.clip(return_val, -0.05, 0.05)
                    close_prices[i] = close_prices[i-1] * (1 + return_val)
                df['close'] = close_prices
            else:
                df['close'] = 50000 + np.cumsum(np.random.normal(0, 50, len(df)))
        
        for col in ['volume', 'bid_qty', 'ask_qty', 'buy_qty', 'sell_qty']:
            if col not in df.columns:
                df[col] = np.random.exponential(1000 if col == 'volume' else 100, len(df))
        
        if 'high' not in df.columns:
            df['high'] = df['close'] * 1.002
        if 'low' not in df.columns:
            df['low'] = df['close'] * 0.998
    
    # Feature selection (top 50 for efficiency)
    x_features = [col for col in train_df.columns if col.startswith('X') and col[1:].isdigit()]
    if len(x_features) > 50:
        logger.info(f"Selecting top 50 features from {len(x_features)} proprietary features...")
        correlations = []
        for feature in x_features:
            if 'label' in train_df.columns:
                corr = abs(train_df[feature].corr(train_df['label']))
                correlations.append((feature, corr if not pd.isna(corr) else 0))
        
        correlations.sort(key=lambda x: x[1], reverse=True)
        selected_features = [feat for feat, corr in correlations[:50]]
        
        for i, feature in enumerate(selected_features):
            train_df[f'selected_X{i+1}'] = train_df[feature]
            test_df[f'selected_X{i+1}'] = test_df[feature]
        
        logger.info(f"Selected {len(selected_features)} best features")
    
    # Clean data
    train_df = train_df.fillna(0)
    test_df = test_df.fillna(0)
    if 'label' in train_df.columns:
        train_df['label'] = np.clip(train_df['label'], -0.05, 0.05)
    
    # Initialize correlation-optimized MDP
    mdp = CorrelationOptimizedMDP(
        lookback_window=100,
        prediction_horizon=1,
        use_proprietary_features=True,
        risk_free_rate=0.02,
        transaction_cost=0.001,
        max_position=1.0,
        cvar_alpha=0.05,
        risk_tolerance=0.1
    )
    
    # Get dimensions
    dummy_state = mdp.reset(train_df.iloc[:mdp.lookback_window])
    state_dim = len(dummy_state)
    action_dim = mdp.action_dim
    
    logger.info(f"MDP initialized - State dim: {state_dim}, Action dim: {action_dim}")
    
    # Setup device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    # Create correlation-optimized ensemble
    ensemble_agent = create_correlation_optimized_ensemble(state_dim, action_dim, device, logger)
    
    # Training with correlation focus
    logger.info("Starting correlation-optimized training...")
    
    # Use subset for efficient training (100K samples)
    training_samples = min(100000, len(train_df))
    train_subset = train_df.iloc[:training_samples].copy()
    
    logger.info(f"Training on {len(train_subset):,} samples for correlation optimization")
    
    # Training loop
    replay_buffer = ReplayBuffer(capacity=50000)
    state = mdp.reset(train_subset.iloc[:mdp.lookback_window])
    
    correlations = []
    
    for idx in range(mdp.lookback_window, len(train_subset)):
        if idx % 5000 == 0:
            logger.info(f"Training step {idx:,}/{len(train_subset):,}")
            
            # Log current correlation if available
            if hasattr(mdp, 'prediction_history') and len(mdp.prediction_history) >= 100:
                current_corr = np.corrcoef(mdp.prediction_history, mdp.actual_history)[0, 1]
                if not np.isnan(current_corr):
                    correlations.append(current_corr)
                    logger.info(f"  Current correlation: {current_corr:.4f}")
        
        current_row = train_subset.iloc[idx]
        
        # Get ensemble prediction
        action = ensemble_agent.select_action(state, deterministic=False)
        
        # Step environment
        next_state, reward, done, info = mdp.step(action, current_row)
        
        # Store transition
        replay_buffer.add(state, action, reward, next_state, done)
        
        # Update agents
        if len(replay_buffer) >= 1000 and idx % 10 == 0:
            try:
                batch = replay_buffer.sample(64, device=ensemble_agent.device)
                losses = ensemble_agent.update(batch)
            except Exception as e:
                logger.warning(f"Update failed at step {idx}: {e}")
        
        # Update state
        if done:
            state = mdp.reset(train_subset.iloc[max(0, idx-mdp.lookback_window):idx])
        else:
            state = next_state
    
    # Final correlation
    if correlations:
        final_corr = correlations[-1]
        logger.info(f"Final training correlation: {final_corr:.4f}")
    
    # Generate test predictions
    logger.info("Generating correlation-optimized predictions...")
    
    predictions = []
    state = mdp.reset(test_df.iloc[:mdp.lookback_window])
    
    with torch.no_grad():
        for idx in range(len(test_df)):
            if idx % 25000 == 0:
                logger.info(f"Prediction progress: {idx:,}/{len(test_df):,}")
            
            current_row = test_df.iloc[idx]
            
            if idx < mdp.lookback_window:
                prediction = 0.0
            else:
                prediction = ensemble_agent.select_action(state, deterministic=True)
                next_state, _, done, _ = mdp.step(prediction, current_row)
                
                if done:
                    state = mdp.reset(test_df.iloc[max(0, idx-mdp.lookback_window):idx])
                else:
                    state = next_state
            
            predictions.append({
                'ID': current_row.get('ID', idx),
                'prediction': float(prediction)
            })
    
    # Save predictions
    pred_df = pd.DataFrame(predictions)
    pred_df['ID'] = pred_df['ID'].astype(int)
    pred_df = pred_df.sort_values('ID').reset_index(drop=True)
    
    # Save in multiple formats
    csv_path = os.path.join(output_dir, 'correlation_optimized_submission.csv')
    pred_df.to_csv(csv_path, index=False)
    
    gz_path = os.path.join(output_dir, 'correlation_optimized_submission.csv.gz')
    pred_df.to_csv(gz_path, index=False, compression='gzip')
    
    import zipfile
    zip_path = os.path.join(output_dir, 'correlation_optimized_submission.zip')
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(csv_path, 'correlation_optimized_submission.csv')
    
    # Save model
    model_path = os.path.join(output_dir, 'correlation_optimized_model')
    ensemble_agent.save(model_path)
    
    logger.info("="*80)
    logger.info("CORRELATION-OPTIMIZED TRAINING COMPLETE!")
    logger.info(f"Generated {len(pred_df):,} predictions")
    logger.info(f"Prediction statistics:")
    logger.info(f"  Mean: {pred_df['prediction'].mean():.6f}")
    logger.info(f"  Std:  {pred_df['prediction'].std():.6f}")
    logger.info(f"  Range: [{pred_df['prediction'].min():.6f}, {pred_df['prediction'].max():.6f}]")
    if correlations:
        logger.info(f"Final training correlation: {final_corr:.4f}")
    logger.info("="*80)
    
    return pred_df

def main():
    """Main correlation-optimized pipeline"""
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f'outputs/correlation_optimized_{timestamp}'
    logger = setup_logging(output_dir)
    
    logger.info("="*80)
    logger.info("PEARSON CORRELATION OPTIMIZED CRYPTO RL PIPELINE")
    logger.info("="*80)
    
    try:
        predictions_df = correlation_optimized_training(
            'train.parquet', 
            'test.parquet', 
            output_dir, 
            logger
        )
        
        logger.info("🎯 CORRELATION-OPTIMIZED SUBMISSION READY!")
        logger.info(f"📁 Files available in: {output_dir}")
        logger.info("📄 Upload correlation_optimized_submission.csv/.gz/.zip")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    main()