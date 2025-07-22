import argparse
import yaml
import logging
import sys
import os
from datetime import datetime
import torch
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents import SACAgent, PPOAgent, DDPGAgent, TD3Agent, TDMPC2Agent, EnsembleAgent
from trainers import WalkForwardTrainer
from utils.data_loader import CryptoDataLoader
from utils.config import load_config

# Import MDP from crypto_mdp
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from crypto_mdp.mdp import CryptoPricePredictionMDP

def setup_logging(log_dir: str):
    """Setup logging configuration"""
    os.makedirs(log_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'training.log')),
            logging.StreamHandler()
        ]
    )

def main():
    parser = argparse.ArgumentParser(description='Train RL agent for crypto price prediction')
    parser.add_argument('--config', type=str, default='configs/default_config.yaml', help='Config file path')
    parser.add_argument('--data', type=str, required=True, help='Path to training data (parquet)')
    parser.add_argument('--agent', type=str, default='sac', choices=['sac', 'ppo', 'ddpg', 'td3', 'td_mpc2', 'ensemble'])
    parser.add_argument('--output', type=str, default='outputs', help='Output directory')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Setup output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(args.output, f'{args.agent}_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup logging
    setup_logging(output_dir)
    logger = logging.getLogger(__name__)
    
    logger.info(f"Starting training with agent: {args.agent}")
    logger.info(f"Output directory: {output_dir}")
    
    # Load configuration
    config = load_config(args.config)
    
    # Initialize MDP
    mdp_config = config.get('mdp', {})
    mdp = CryptoPricePredictionMDP(**mdp_config)
    
    # Load initial data to get state dimension
    logger.info(f"Loading data from {args.data}")
    data_loader = CryptoDataLoader(args.data)
    
    # Get a sample batch to initialize dimensions
    sample_batch = next(data_loader.iterate_batches())
    sample_state = mdp.reset(sample_batch.iloc[:100])
    state_dim = len(sample_state)
    action_dim = mdp.action_dim
    
    logger.info(f"State dimension: {state_dim}")
    logger.info(f"Action dimension: {action_dim}")
    
    # Initialize agent
    agent_config = config.get(f'{args.agent}_config', {})
    
    if args.agent == 'sac':
        agent = SACAgent(state_dim, action_dim, agent_config, args.device)
    elif args.agent == 'ppo':
        agent = PPOAgent(state_dim, action_dim, agent_config, args.device)
    elif args.agent == 'ddpg':
        agent = DDPGAgent(state_dim, action_dim, agent_config, args.device)
    elif args.agent == 'td3':
        agent = TD3Agent(state_dim, action_dim, agent_config, args.device)
    elif args.agent == 'td_mpc2':
        agent = TDMPC2Agent(state_dim, action_dim, agent_config, args.device)
    elif args.agent == 'ensemble':
        agent = EnsembleAgent(state_dim, action_dim, agent_config, args.device)
    
    # Initialize trainer
    trainer_config = config.get('trainer', {})
    trainer = WalkForwardTrainer(agent, mdp, trainer_config)
    
    # Train
    logger.info("Starting walk-forward training...")
    
    # For large files, we'll process in chunks
    all_data = []
    for batch_df in data_loader.iterate_batches():
        all_data.append(batch_df)
        
        # Train when we have enough data
        if len(all_data) * len(batch_df) >= trainer_config.get('min_train_size', 1000000):
            combined_df = pd.concat(all_data, ignore_index=True)
            results = trainer.train(combined_df, num_epochs=config.get('num_epochs', 1))
            
            # Save intermediate results
            trainer.save_predictions(os.path.join(output_dir, f'predictions_chunk_{len(all_data)}.csv'))
            
            # Clear processed data
            all_data = all_data[-10:]  # Keep last 10 batches for continuity
    
    # Final training on remaining data
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        results = trainer.train(combined_df, num_epochs=config.get('num_epochs', 1))
    
    # Save final model
    agent.save(os.path.join(output_dir, 'final_model.pth'))
    
    # Save results
    import json
    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(results['final_metrics'], f, indent=4)
    
    logger.info("Training completed!")
    logger.info(f"Results saved to {output_dir}")

if __name__ == "__main__":
    main()
