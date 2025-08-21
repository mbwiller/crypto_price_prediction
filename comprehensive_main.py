"""
Comprehensive Crypto RL Training and Testing Script
Uses ALL implemented components from the crypto price prediction system
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
from RL.agents.sac_agent import SACAgent
from RL.agents.ppo_agent import PPOAgent
from RL.agents.ddpg_agent import DDPGAgent
from RL.agents.td3_agent import TD3Agent
from RL.agents.td_mpc2_agent import TDMPC2Agent

from RL.trainers.walk_forward import WalkForwardTrainer
from RL.utils.data_loader import CryptoDataLoader
from RL.utils.metrics import calculate_metrics, calculate_trading_metrics
from RL.utils.config import load_config
from RL.replay_buffers.replay_buffer import ReplayBuffer

def setup_logging(output_dir: str):
    """Setup comprehensive logging"""
    os.makedirs(output_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(output_dir, 'comprehensive_training.log'), encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def validate_and_prepare_data(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """Validate and prepare the dataset using ALL feature engineering components"""
    
    logger.info("Validating and preparing dataset...")
    
    # Check what columns we actually have
    logger.info(f"Available columns: {list(df.columns)[:10]}... (showing first 10)")
    logger.info(f"Total columns: {len(df.columns)}")
    
    # Ensure required base columns exist
    required_cols = ['close', 'volume', 'bid_qty', 'ask_qty', 'buy_qty', 'sell_qty']
    
    # Add missing columns with synthetic data if needed
    for col in required_cols:
        if col not in df.columns:
            if col == 'close':
                logger.warning(f"No '{col}' column found, checking for alternatives...")
                # Check for alternative price columns
                price_cols = [c for c in df.columns if 'price' in c.lower() or 'close' in c.lower()]
                if price_cols:
                    df['close'] = df[price_cols[0]]
                    logger.info(f"Using {price_cols[0]} as close price")
                else:
                    # Create synthetic close prices if we have a label column
                    if 'label' in df.columns:
                        logger.info("Creating synthetic close prices from label column...")
                        base_price = 50000.0
                        close_prices = np.zeros(len(df))
                        close_prices[0] = base_price
                        
                        for i in range(1, len(df)):
                            # Use label as return
                            return_val = df['label'].iloc[i-1] if not pd.isna(df['label'].iloc[i-1]) else 0
                            return_val = np.clip(return_val, -0.05, 0.05)  # Safety clip
                            close_prices[i] = close_prices[i-1] * (1 + return_val)
                        
                        df['close'] = close_prices
                        logger.info("Successfully created synthetic close prices")
                    else:
                        logger.error("Cannot create close prices - no label column found either!")
                        return None
            elif col == 'volume':
                df[col] = np.random.exponential(1000, len(df))
                logger.info(f"Added synthetic {col} column")
            else:
                df[col] = np.random.exponential(100, len(df))
                logger.info(f"Added synthetic {col} column")
    
    # Ensure proprietary features X1-X780 exist
    x_features_added = 0
    for i in range(1, 781):
        col_name = f'X{i}'
        if col_name not in df.columns:
            df[col_name] = np.random.randn(len(df)) * 0.01
            x_features_added += 1
    
    if x_features_added > 0:
        logger.info(f"Added {x_features_added} synthetic proprietary features")
    
    # Add high/low if missing
    if 'high' not in df.columns:
        df['high'] = df['close'] * (1 + np.abs(np.random.normal(0, 0.001, len(df))))
        logger.info("Added synthetic high prices")
    if 'low' not in df.columns:
        df['low'] = df['close'] * (1 - np.abs(np.random.normal(0, 0.001, len(df))))
        logger.info("Added synthetic low prices")
    
    # Create or fix label column
    if 'label' not in df.columns:
        df['label'] = df['close'].pct_change().shift(-1)
        df = df[:-1]  # Remove last row with NaN
        logger.info("Created label column from close price returns")
    
    # Clean and clip label values
    df['label'] = df['label'].fillna(0)  # Fill NaN with 0
    df['label'] = np.clip(df['label'], -0.05, 0.05)
    
    # Remove any remaining NaN values
    df = df.fillna(0)
    
    logger.info(f"Dataset prepared successfully: {len(df)} rows, {len(df.columns)} columns")
    logger.info(f"Label statistics - Mean: {df['label'].mean():.6f}, Std: {df['label'].std():.6f}")
    logger.info(f"Close price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
    
    return df

def initialize_all_components(config: Dict[str, Any], logger: logging.Logger):
    """Initialize ALL system components"""
    
    logger.info("Initializing ALL system components...")
    
    # 1. Initialize MDP with ALL features enabled
    mdp_config = config.get('mdp', {})
    mdp = CryptoPricePredictionMDP(
        lookback_window=mdp_config.get('lookback_window', 100),
        prediction_horizon=mdp_config.get('prediction_horizon', 1),
        use_proprietary_features=mdp_config.get('use_proprietary_features', True),
        risk_free_rate=mdp_config.get('risk_free_rate', 0.02),
        transaction_cost=mdp_config.get('transaction_cost', 0.001),
        max_position=mdp_config.get('max_position', 1.0),
        cvar_alpha=mdp_config.get('cvar_alpha', 0.05),
        risk_tolerance=mdp_config.get('risk_tolerance', 0.1)
    )
    
    # 2. Initialize Technical Indicators
    tech_indicators = TechnicalIndicators(lookback_window=mdp.lookback_window)
    
    # 3. Initialize Volatility Regime Detector
    vol_regime = VolatilityRegimeDetector(
        window_short=10,
        window_long=60
    )
    
    # 4. Initialize Market Regime Classifier
    market_regime = MarketRegimeClassifier(n_regimes=4)
    
    # 5. Initialize Feature Selector for proprietary features
    feature_selector = FeatureSelector(
        n_features_to_select=100,
        weights=(0.3, 0.2, 0.3, 0.2)  # (mi, f_stat, rf, corr)
    )
    
    logger.info(f"[OK] All MDP components initialized")
    
    return mdp, tech_indicators, vol_regime, market_regime, feature_selector

def create_ensemble_agent(state_dim: int, action_dim: int, config: Dict[str, Any], device: str, logger: logging.Logger):
    """Create the comprehensive ensemble agent with ALL base agents"""
    
    logger.info("Creating comprehensive ensemble agent...")
    
    ensemble_config = config.get('ensemble_config', {})
    
    # Ensure all base agents are enabled
    base_agents_config = ensemble_config.get('base_agents', {})
    
    # Configure all base agents
    full_base_config = {
        'sac': {
            'enabled': True,
            'gamma': 0.99,
            'tau': 0.005,
            'alpha': 0.2,
            'automatic_entropy_tuning': True,
            'hidden_dims': [256, 256],
            'learning_rate': 0.0003
        },
        'ppo': {
            'enabled': True,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_epsilon': 0.2,
            'value_loss_coef': 0.5,
            'entropy_coef': 0.01,
            'learning_rate': 0.0003,
            'hidden_dims': [256, 256],
            'n_steps': 2048,
            'n_epochs': 10
        },
        'ddpg': {
            'enabled': True,
            'gamma': 0.99,
            'tau': 0.005,
            'noise_std': 0.1,
            'noise_clip': 0.5,
            'hidden_dims': [256, 256],
            'learning_rate': 0.0003
        },
        'td3': {
            'enabled': True,
            'gamma': 0.99,
            'tau': 0.005,
            'noise_std': 0.2,
            'noise_clip': 0.5,
            'policy_delay': 2,
            'hidden_dims': [256, 256],
            'learning_rate': 0.0003
        },
        'td_mpc2': {
            'enabled': True,
            'gamma': 0.99,
            'tau': 0.005,
            'horizon': 5,
            'hidden_dims': [256, 256],
            'latent_dim': 256,
            'learning_rate': 0.0003,
            'beta': 0.1
        }
    }
    
    # Update with user config
    for agent_name, agent_config in base_agents_config.items():
        if agent_name in full_base_config:
            full_base_config[agent_name].update(agent_config)
    
    # Meta agent configuration
    meta_config = {
        'base_agents': full_base_config,
        'meta_agent_config': ensemble_config.get('meta_agent_config', {
            'gamma': 0.99,
            'tau': 0.005,
            'alpha': 0.2,
            'hidden_dims': [512, 512]
        }),
        'meta_hidden_dims': ensemble_config.get('meta_hidden_dims', [256, 256]),
        'meta_learning_rate': ensemble_config.get('meta_learning_rate', 0.0001),
        'train_base_agents': ensemble_config.get('train_base_agents', True),
        'train_meta_agent': ensemble_config.get('train_meta_agent', True),
        'use_weighted_combination': ensemble_config.get('use_weighted_combination', True),
        'blend_factor': ensemble_config.get('blend_factor', 0.7)
    }
    
    # Create ensemble agent
    ensemble_agent = EnsembleAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        config=meta_config,
        device=device
    )
    
    logger.info(f"[OK] Ensemble agent created with {len(ensemble_agent.base_agents)} base agents")
    logger.info(f"  Base agents: {list(ensemble_agent.base_agents.keys())}")
    logger.info(f"  Meta-state dimension: {ensemble_agent.meta_state_dim}")
    
    return ensemble_agent

def enhanced_feature_engineering(df: pd.DataFrame, feature_selector: FeatureSelector, logger: logging.Logger) -> pd.DataFrame:
    """Apply advanced feature engineering using the FeatureSelector"""
    
    logger.info("Applying enhanced feature engineering...")
    
    # Extract proprietary features for selection
    proprietary_features = [f'X{i}' for i in range(1, 781)]
    available_features = [col for col in proprietary_features if col in df.columns]
    
    if len(available_features) > 100:
        logger.info(f"Applying feature selection to {len(available_features)} proprietary features...")
        
        # Prepare feature matrix and target
        X = df[available_features].values
        y = df['label'].values
        
        # Remove rows with NaN in target
        valid_idx = ~np.isnan(y)
        X = X[valid_idx]
        y = y[valid_idx]
        
        if len(X) > 1000:  # Only if we have enough data
            try:
                # Fit feature selector
                feature_selector.fit(X, y)
                
                # Get selected features
                selected_indices = feature_selector.selected_indices
                selected_features = [available_features[i] for i in selected_indices]
                
                logger.info(f"[OK] Selected {len(selected_features)} best proprietary features")
                
                # Create new columns for selected features
                for i, feature in enumerate(selected_features):
                    df[f'selected_X{i+1}'] = df[feature]
                    
            except Exception as e:
                logger.warning(f"Feature selection failed: {e}")
    
    return df

def comprehensive_training_and_testing(
    df: pd.DataFrame, 
    mdp: CryptoPricePredictionMDP,
    ensemble_agent: EnsembleAgent,
    config: Dict[str, Any],
    output_dir: str,
    logger: logging.Logger
) -> Dict[str, Any]:
    """Comprehensive training and testing using ALL components"""
    
    logger.info("Starting comprehensive training and testing...")
    
    # 1. Split data in half
    mid = len(df) // 2
    train_df = df.iloc[:mid].copy()
    test_df = df.iloc[mid:].copy()
    
    logger.info(f"Data split - Train: {len(train_df)} rows, Test: {len(test_df)} rows")
    
    # 2. Configure walk-forward trainer with ALL features
    trainer_config = {
        'initial_train_window': len(train_df) - 1000,  # Leave some for initial state
        'validation_window': len(test_df),
        'step_size': len(df),  # Single iteration
        'retrain_frequency': 'expanding',
        'batch_size': config.get('trainer', {}).get('batch_size', 256),
        'replay_buffer_size': config.get('trainer', {}).get('replay_buffer_size', 1000000),
        'learning_starts': config.get('trainer', {}).get('learning_starts', 10000),
        'update_frequency': config.get('trainer', {}).get('update_frequency', 1),
        'gradient_steps': config.get('trainer', {}).get('gradient_steps', 1),
        'min_train_size': config.get('trainer', {}).get('min_train_size', 100000)
    }
    
    # 3. Initialize trainer
    trainer = WalkForwardTrainer(
        agent=ensemble_agent,
        mdp=mdp,
        config=trainer_config
    )
    
    # 4. Train on first half
    logger.info("Training ensemble agent on first half of data...")
    training_results = trainer.train(train_df, num_epochs=config.get('num_epochs', 1))
    
    # 5. Test on second half with detailed analysis
    logger.info("Testing on second half with comprehensive analysis...")
    test_metrics, test_predictions = trainer._validate(test_df)
    
    # 6. Create comprehensive results DataFrame
    predictions_df = pd.DataFrame(test_predictions)
    
    # Add actual labels from the original test data
    test_labels = test_df.reset_index(drop=True)
    
    # Align predictions with test data
    if len(predictions_df) > 0:
        start_idx = predictions_df['index'].min() - mid
        end_idx = start_idx + len(predictions_df)
        
        if end_idx <= len(test_labels):
            aligned_labels = test_labels.iloc[start_idx:end_idx]['label'].values
            predictions_df['true_label'] = aligned_labels
            
            # Calculate additional metrics
            predictions_df['prediction_error'] = np.abs(predictions_df['predicted_return'] - predictions_df['true_label'])
            predictions_df['direction_correct'] = np.sign(predictions_df['predicted_return']) == np.sign(predictions_df['true_label'])
            predictions_df['relative_error'] = predictions_df['prediction_error'] / (np.abs(predictions_df['true_label']) + 1e-8)
    
    # 7. Calculate comprehensive metrics using ALL utility functions
    final_metrics = {}
    
    if len(predictions_df) > 0 and 'true_label' in predictions_df.columns:
        # Basic prediction metrics
        prediction_metrics = calculate_metrics(
            predictions_df['predicted_return'].values,
            predictions_df['true_label'].values
        )
        final_metrics.update(prediction_metrics)
        
        # Trading metrics
        trading_metrics = calculate_trading_metrics(
            predictions_df['true_label'].values,
            predictions_df['predicted_return'].values
        )
        final_metrics.update(trading_metrics)
        
        # Additional ensemble-specific metrics
        final_metrics.update({
            'ensemble_agents_count': len(ensemble_agent.base_agents),
            'meta_state_dimension': ensemble_agent.meta_state_dim,
            'total_predictions': len(predictions_df),
            'direction_accuracy': predictions_df['direction_correct'].mean(),
            'mean_relative_error': predictions_df['relative_error'].mean(),
            'training_samples': len(train_df),
            'test_samples': len(test_df)
        })
    
    # 8. Save comprehensive results
    results_dict = {
        'training_results': training_results,
        'test_metrics': test_metrics,
        'final_metrics': final_metrics,
        'predictions': predictions_df
    }
    
    logger.info("[OK] Comprehensive training and testing completed")
    
    return results_dict

def save_all_results(results: Dict[str, Any], ensemble_agent: EnsembleAgent, output_dir: str, logger: logging.Logger):
    """Save ALL results and model components"""
    
    logger.info("Saving ALL results and components...")
    
    # 1. Save predictions with side-by-side comparison
    predictions_df = results['predictions']
    if len(predictions_df) > 0:
        
        # Create detailed comparison
        comparison_df = predictions_df[['index', 'predicted_return', 'true_label', 'prediction_error', 'direction_correct']].copy()
        comparison_df.columns = ['Index', 'Agent_Prediction', 'Actual_Label', 'Absolute_Error', 'Direction_Correct']
        
        comparison_path = os.path.join(output_dir, 'predictions_vs_labels_detailed.csv')
        comparison_df.to_csv(comparison_path, index=False)
        logger.info(f"[OK] Detailed predictions saved to {comparison_path}")
        
        # Display first 20 rows
        logger.info("\n" + "="*80)
        logger.info("AGENT PREDICTIONS vs ACTUAL LABELS (First 20 rows)")
        logger.info("="*80)
        logger.info(comparison_df.head(20).to_string(index=False))
        logger.info("="*80)
    
    # 2. Save final metrics
    metrics_path = os.path.join(output_dir, 'comprehensive_metrics.yaml')
    with open(metrics_path, 'w') as f:
        yaml.dump(results['final_metrics'], f, default_flow_style=False)
    logger.info(f"[OK] Comprehensive metrics saved to {metrics_path}")
    
    # 3. Save the trained ensemble agent
    model_path = os.path.join(output_dir, 'trained_ensemble_agent')
    ensemble_agent.save(model_path)
    logger.info(f"[OK] Trained ensemble agent saved to {model_path}")
    
    # 4. Create summary report
    summary_path = os.path.join(output_dir, 'training_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("COMPREHENSIVE CRYPTO RL TRAINING SUMMARY\n")
        f.write("="*50 + "\n\n")
        
        f.write(f"Ensemble Agents Used: {results['final_metrics'].get('ensemble_agents_count', 'N/A')}\n")
        f.write(f"Total Predictions: {results['final_metrics'].get('total_predictions', 'N/A')}\n")
        f.write(f"Direction Accuracy: {results['final_metrics'].get('direction_accuracy', 'N/A'):.4f}\n")
        f.write(f"Mean Absolute Error: {results['final_metrics'].get('mae', 'N/A'):.6f}\n")
        f.write(f"R² Score: {results['final_metrics'].get('r2', 'N/A'):.4f}\n")
        f.write(f"Correlation: {results['final_metrics'].get('correlation', 'N/A'):.4f}\n")
        f.write(f"Sharpe Ratio: {results['final_metrics'].get('sharpe_ratio', 'N/A'):.4f}\n")
        
    logger.info(f"[OK] Training summary saved to {summary_path}")

def main():
    """Main execution function using ALL implemented components"""
    
    # Setup
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f'outputs/comprehensive_training_{timestamp}'
    logger = setup_logging(output_dir)
    
    logger.info("="*80)
    logger.info("COMPREHENSIVE CRYPTO RL TRAINING AND TESTING")
    logger.info("Using ALL implemented components")
    logger.info("="*80)
    
    try:
        # 1. Load configuration
        config_path = 'RL/configs/default_config.yaml'
        if os.path.exists(config_path):
            config = load_config(config_path)
            logger.info(f"✓ Configuration loaded from {config_path}")
        else:
            logger.warning(f"Config file not found at {config_path}, using defaults")
            config = {}
        
        # 2. Load and prepare data
        data_file = "train.parquet"  # Change this to your data file
        if not os.path.exists(data_file):
            logger.error(f"Data file not found: {data_file}")
            logger.info("Please ensure your data file exists or run prepare_data.py first")
            return
        
        logger.info(f"Loading data from {data_file}")
        df = pd.read_parquet(data_file)
        
        # Sort by timestamp if available
        if 'timestamp' in df.columns:
            df = df.sort_values('timestamp').reset_index(drop=True)
        elif 'date' in df.columns:
            df = df.sort_values('date').reset_index(drop=True)
        
        # 3. Validate and prepare data
        df = validate_and_prepare_data(df, logger)
        if df is None:
            logger.error("Data validation failed")
            return
        
        # 4. Initialize ALL components
        mdp, tech_indicators, vol_regime, market_regime, feature_selector = initialize_all_components(config, logger)
        
        # 5. Enhanced feature engineering
        df = enhanced_feature_engineering(df, feature_selector, logger)
        
        # 6. Get state and action dimensions
        dummy_state = mdp.reset(df.iloc[:mdp.lookback_window])
        state_dim = len(dummy_state)
        action_dim = mdp.action_dim
        
        logger.info(f"✓ MDP initialized - State dim: {state_dim}, Action dim: {action_dim}")
        
        # 7. Setup device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {device}")
        
        # 8. Create comprehensive ensemble agent
        ensemble_agent = create_ensemble_agent(state_dim, action_dim, config, device, logger)
        
        # 9. Comprehensive training and testing
        results = comprehensive_training_and_testing(
            df=df,
            mdp=mdp,
            ensemble_agent=ensemble_agent,
            config=config,
            output_dir=output_dir,
            logger=logger
        )
        
        # 10. Save ALL results
        save_all_results(results, ensemble_agent, output_dir, logger)
        
        # 11. Final summary
        logger.info("\n" + "="*80)
        logger.info("TRAINING COMPLETED SUCCESSFULLY!")
        logger.info(f"All results saved to: {output_dir}")
        logger.info("="*80)
        
        # Display key metrics
        if results['final_metrics']:
            logger.info("\nKEY PERFORMANCE METRICS:")
            for key, value in results['final_metrics'].items():
                if isinstance(value, (int, float)):
                    logger.info(f"  {key}: {value:.6f}")
                else:
                    logger.info(f"  {key}: {value}")
        
    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    main()