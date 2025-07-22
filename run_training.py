"""
Main training script for crypto price prediction
"""

import os
import sys
import argparse
import logging
import torch
import numpy as np
from datetime import datetime

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from RL.scripts.train import main as train_main

def setup_environment():
    """Setup compute environment for your node"""
    # Set number of threads for CPU operations
    torch.set_num_threads(10)  # Use all 10 cores
    
    # Set memory limits
    os.environ['OMP_NUM_THREADS'] = '10'
    os.environ['MKL_NUM_THREADS'] = '10'
    
    # Check CUDA availability
    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.device_count()} devices")
        device = 'cuda'
    else:
        print("CUDA not available, using CPU")
        device = 'cpu'
    
    return device

def prepare_data(data_path):
    """Prepare and validate data"""
    import pandas as pd
    
    if data_path.endswith('.csv'):
        # Load your CSV data
        df = pd.read_csv(data_path)
        
        # Ensure required columns exist
        required_cols = ['close', 'volume', 'bid_qty', 'ask_qty', 'buy_qty', 'sell_qty']
        required_cols.extend([f'X{i}' for i in range(1, 781)])  # Proprietary features
        
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            print(f"Warning: Missing columns: {missing_cols}")
            # Add dummy columns if needed
            for col in missing_cols:
                if col.startswith('X'):
                    df[col] = np.random.randn(len(df)) * 0.01
                else:
                    df[col] = 0
        
        # Add label column if missing (next minute return)
        if 'label' not in df.columns:
            df['label'] = df['close'].pct_change().shift(-1)
            df = df[:-1]  # Remove last row with NaN label
        
        # Save as parquet for faster loading
        output_path = data_path.replace('.csv', '.parquet')
        df.to_parquet(output_path, index=False)
        return output_path
    
    return data_path

def run_training(args):
    """Main training execution"""
    # Setup environment
    device = setup_environment()
    
    # Prepare data
    data_path = prepare_data(args.data)
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(args.output, f'{args.agent}_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(output_dir, 'training.log')),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Starting training with {args.agent} on {device}")
    logger.info(f"Data: {data_path}")
    logger.info(f"Output: {output_dir}")
    
    # Prepare arguments for train.py
    train_args = [
        '--config', args.config,
        '--data', data_path,
        '--agent', args.agent,
        '--output', args.output,
        '--device', device,
        '--seed', str(args.seed)
    ]
    
    if args.resume:
        train_args.extend(['--resume', args.resume])
    
    # Mock argparse for train.py
    import sys
    sys.argv = ['train.py'] + train_args
    
    # Run training
    train_main()

def main():
    parser = argparse.ArgumentParser(description='Run crypto RL training')
    parser.add_argument('--data', type=str, required=True, 
                       help='Path to training data (CSV or parquet)')
    parser.add_argument('--agent', type=str, default='ensemble',
                       choices=['sac', 'ppo', 'ddpg', 'td3', 'td_mpc2', 'ensemble'],
                       help='RL agent to use')
    parser.add_argument('--config', type=str, 
                       default='RL/configs/default_config.yaml',
                       help='Configuration file')
    parser.add_argument('--output', type=str, default='outputs',
                       help='Output directory')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from checkpoint')
    
    args = parser.parse_args()
    
    run_training(args)

if __name__ == '__main__':
    main()
