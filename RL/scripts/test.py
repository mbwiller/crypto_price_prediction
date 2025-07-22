import argparse
import torch
import numpy as np
import pandas as pd
import logging
import sys
import os
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents import SACAgent, EnsembleAgent
from utils.data_loader import CryptoDataLoader
from utils.metrics import calculate_metrics

# Import MDP
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from crypto_mdp.mdp import CryptoPricePredictionMDP

def main():
    parser = argparse.ArgumentParser(description='Test trained RL agent')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model')
    parser.add_argument('--data', type=str, required=True, help='Path to test data (parquet)')
    parser.add_argument('--output', type=str, required=True, help='Output predictions file')
    parser.add_argument('--agent', type=str, default='sac', choices=['sac', 'ensemble'])
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Load test data
    logger.info(f"Loading test data from {args.data}")
    data_loader = CryptoDataLoader(args.data)
    
    # Initialize MDP
    mdp = CryptoPricePredictionMDP()
    
    # Get dimensions
    sample_batch = next(data_loader.iterate_batches())
    sample_state = mdp.reset(sample_batch.iloc[:100])
    state_dim = len(sample_state)
    action_dim = mdp.action_dim
    
    # Load agent
    if args.agent == 'sac':
        agent = SACAgent(state_dim, action_dim, {}, args.device)
    elif args.agent == 'ensemble':
        agent = EnsembleAgent(state_dim, action_dim, {}, args.device)
    
    agent.load(args.model)
    logger.info(f"Loaded model from {args.model}")
    
    # Generate predictions
    predictions = []
    
    for batch_df in tqdm(data_loader.iterate_batches(), desc="Generating predictions"):
        # Reset MDP
        state = mdp.reset(batch_df.iloc[:100])
        
        # Generate predictions for batch
        for idx in range(100, len(batch_df)):
            current_row = batch_df.iloc[idx]
            
            # Get prediction
            with torch.no_grad():
                action = agent.select_action(state, deterministic=True)
            
            # Store prediction
            predictions.append({
                'ID': int(current_row.get('ID', idx)),
                'prediction': float(action)
            })
            
            # Step environment
            next_state, _, done, _ = mdp.step(action, current_row)
            
            if done:
                state = mdp.reset(batch_df.iloc[max(0, idx-100):idx])
            else:
                state = next_state
    
    # Save predictions
    pred_df = pd.DataFrame(predictions)
    pred_df.to_csv(args.output, index=False)
    logger.info(f"Saved {len(pred_df)} predictions to {args.output}")

if __name__ == "__main__":
    main()
