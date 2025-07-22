"""Test script to verify setup"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test all required imports"""
    print("Testing imports...")
    
    try:
        # Crypto MDP imports
        from crypto_MDP.mdp import CryptoPricePredictionMDP
        from crypto_MDP.indicators import TechnicalIndicators
        from crypto_MDP.regime import VolatilityRegimeDetector
        print("✓ Crypto MDP modules imported successfully")
        
        # RL imports
        from RL.agents import SACAgent, EnsembleAgent
        from RL.trainers.walk_forward import WalkForwardTrainer
        from RL.utils.data_loader import CryptoDataLoader
        print("✓ RL modules imported successfully")
        
        # External dependencies
        import torch
        import pandas as pd
        import numpy as np
        import talib
        print("✓ External dependencies imported successfully")
        
        # Check PyTorch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_mdp_initialization():
    """Test MDP initialization"""
    print("\nTesting MDP initialization...")
    
    try:
        from crypto_MDP.mdp import CryptoPricePredictionMDP
        import pandas as pd
        import numpy as np
        
        # Create dummy data
        n_samples = 1000
        dummy_data = pd.DataFrame({
            'close': np.random.randn(n_samples).cumsum() + 100,
            'volume': np.random.exponential(1000, n_samples),
            'bid_qty': np.random.exponential(100, n_samples),
            'ask_qty': np.random.exponential(100, n_samples),
            'buy_qty': np.random.exponential(50, n_samples),
            'sell_qty': np.random.exponential(50, n_samples),
            **{f'X{i}': np.random.randn(n_samples) for i in range(1, 781)}
        })
        
        # Initialize MDP
        mdp = CryptoPricePredictionMDP()
        state = mdp.reset(dummy_data.iloc[:100])
        
        print(f"✓ MDP initialized successfully")
        print(f"  State dimension: {len(state)}")
        print(f"  Action dimension: {mdp.action_dim}")
        
        return True
        
    except Exception as e:
        print(f"✗ MDP initialization error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    print("=== Crypto RL Setup Test ===\n")
    
    tests_passed = 0
    tests_total = 2
    
    if test_imports():
        tests_passed += 1
    
    if test_mdp_initialization():
        tests_passed += 1
    
    print(f"\n=== Results: {tests_passed}/{tests_total} tests passed ===")
    
    if tests_passed == tests_total:
        print("\n✓ All tests passed! Ready to train.")
    else:
        print("\n✗ Some tests failed. Please fix the issues before training.")
