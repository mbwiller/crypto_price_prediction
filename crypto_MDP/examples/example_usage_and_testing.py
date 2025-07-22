# Example usage and testing
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    
    # Generate synthetic data with proprietary features
    data = pd.DataFrame({
        'bid_qty': np.random.exponential(100, n_samples),
        'ask_qty': np.random.exponential(100, n_samples),
        'buy_qty': np.random.exponential(50, n_samples),
        'sell_qty': np.random.exponential(50, n_samples),
        'volume': np.random.exponential(1000, n_samples),
        'high': 50000 + np.cumsum(np.random.randn(n_samples) * 10),
        'low': 49900 + np.cumsum(np.random.randn(n_samples) * 10),
        'close': 50000 + np.cumsum(np.random.randn(n_samples) * 10),
    })
    
    # Add proprietary features
    for i in range(1, 781):
        data[f'X{i}'] = np.random.randn(n_samples) * np.random.rand()
    
    # Add label (what we're trying to predict)
    data['label'] = data['close'].pct_change().shift(-1)  # Next period return
    
    # Initialize MDP environment
    mdp = CryptoPricePredictionMDP(
        lookback_window=50,
        use_proprietary_features=True,
        risk_tolerance=0.05
    )
    
    # Reset with initial data
    initial_state = mdp.reset(data.iloc[:100])
    print(f"Initial state shape: {initial_state.shape}")
    print(f"State space info: {mdp.get_state_space_info()}")
    print(f"Action space info: {mdp.get_action_space_info()}")
    
    # Run a few steps
    for i in range(100, 110):
        # Random action for testing
        action = np.random.uniform(-0.01, 0.01)  # Predict small price change
        
        next_state, reward, done, info = mdp.step(action, data.iloc[i])
        
        print(f"\nStep {i-100}:")
        print(f"  Action (predicted return): {action:.4f}")
        print(f"  Actual return: {info['actual_return']:.4f}")
        print(f"  Reward: {reward:.4f}")
        print(f"  Prediction error: {info['prediction_error']:.4f}")
        print(f"  Portfolio value: {info['portfolio_value']:.4f}")
        print(f"  Volatility regime: {info['volatility_regime']['is_high_vol']}")
        
        if done:
            print("Episode terminated!")
            break
    
    # Test feature selection
    print("\n\nTesting Feature Selection...")
    X = data[[f'X{i}' for i in range(1, 781)]].values[:-1]  # Remove last row due to label shift
    y = data['label'].values[:-1]
    y = y[~np.isnan(y)]  # Remove NaN values
    X = X[:len(y)]  # Align dimensions
    
    selector = FeatureSelector(n_features_to_select=50)
    selector.fit(X, y)
    
    print(f"Top 10 proprietary features:")
    ranking = selector.get_feature_ranking()
    for i, (idx, score) in enumerate(list(ranking.items())[:10]):
        print(f"  {i+1}. X{idx+1}: score = {score:.4f}")
    
    # Test market regime classification
    print("\n\nTesting Market Regime Classification...")
    returns_data = np.random.randn(500, 30) * 0.01  # 500 samples, 30 time steps each
    volatilities = np.std(returns_data, axis=1)
    
    regime_classifier = MarketRegimeClassifier(n_regimes=4)
    regime_classifier.fit(returns_data, volatilities)
    
    # Predict regime for current market
    current_returns = data['close'].pct_change().iloc[-30:].values
    current_returns = current_returns[~np.isnan(current_returns)]
    current_vol = np.std(current_returns)
    
    current_regime = regime_classifier.predict_regime(current_returns, current_vol)
    regime_info = regime_classifier.get_regime_characteristics(current_regime)
    
    print(f"Current market regime: {current_regime}")
    print(f"Regime type: {regime_info['regime_type']}")

    print(f"Regime characteristics: {regime_info}")
