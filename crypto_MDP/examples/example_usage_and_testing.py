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
    print(f"Regime characteristics: {regime_info}") = np.array(self.returns_history)
            price_features = [
                returns[-1] if len(returns) > 0 else 0,  # Last return
                np.mean(returns[-5:]) if len(returns) >= 5 else 0,  # 5-min return
                np.mean(returns[-15:]) if len(returns) >= 15 else 0,  # 15-min return
                np.mean(returns[-30:]) if len(returns) >= 30 else 0,  # 30-min return
                np.std(returns[-10:]) if len(returns) >= 10 else 0,  # Recent volatility
                stats.skew(returns) if len(returns) > 3 else 0,  # Skewness
                stats.kurtosis(returns) if len(returns) > 3 else 0,  # Kurtosis
            ]
        else:
            price_features = [0] * 7
        state_components.extend(price_features)
        
        # 3. Technical Indicators
        if len(self.price_history) >= 20:
            prices = np.array(self.price_history)
            volumes = np.array(self.volume_history)
            highs = np.array(self.high_history)
            lows = np.array(self.low_history)
            closes = prices  # Using prices as closes
            
            indicators = self.tech_indicators.calculate_indicators(prices, volumes, highs, lows, closes)
            state_components.extend(list(indicators.values()))
        else:
            # Default technical indicators
            state_components.extend([0] * 40)  # Approximate number of indicators
        
        # 4. Volatility Regime Features
        if len(self.returns_history) > 10:
            regime_features = self.vol_regime.detect_regime(np.array(self.returns_history))
            state_components.extend(list(regime_features.values()))
        else:
            state_components.extend([0] * 7)
        
        # 5. Market Indicators
        market_features = self._calculate_market_indicators(current_row)
        state_components.extend(market_features)
        
        # 6. Temporal Features
        temporal_features = self._extract_temporal_features()
        state_components.extend(temporal_features)
        
        # 7. Portfolio State
        portfolio_features = [
            self.current_position / self.max_position,  # Normalized position
            self.portfolio_value,
            self.drawdown,
            len(self.action_history) / 1000.0  # Normalized time
        ]
        state_components.extend(portfolio_features)
        
        # 8. Proprietary Features (if available)
        if self.use_proprietary_features:
            proprietary = self._extract_proprietary_features(current_row)
            state_components.extend(proprietary)
        
        # Convert to numpy array
        state = np.array(state_components, dtype=np.float32)
        
        # Handle any NaN or inf values
        state = np.nan_to_num(state, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Normalize state
        if not self.is_fitted and len(self.price_history) >= self.lookback_window:
            self.state_scaler.fit(state.reshape(1, -1))
            self.is_fitted = True
            self.state_dim = len(state)
        
        if self.is_fitted:
            state = self.state_scaler.transform(state.reshape(1, -1)).flatten()
        
        return state
    
    def _calculate_microstructure(self, row: pd.Series) -> MarketMicrostructure:
        """Calculate market microstructure features"""
        
        bid_qty = row.get('bid_qty', 0)
        ask_qty = row.get('ask_qty', 0)
        buy_qty = row.get('buy_qty', 0)
        sell_qty = row.get('sell_qty', 0)
        volume = row.get('volume', 0)
        
        # Bid-ask spread (normalized)
        current_price = self.price_history[-1] if self.price_history else volume
        spread = 2 * (ask_qty - bid_qty) / (ask_qty + bid_qty + 1e-8)
        
        # Order imbalances
        bid_ask_imbalance = (bid_qty - ask_qty) / (bid_qty + ask_qty + 1e-8)
        order_flow_imbalance = (buy_qty - sell_qty) / (buy_qty + sell_qty + 1e-8)
        
        # Kyle's Lambda (price impact)
        if len(self.returns_history) > 0 and volume > 0:
            kyle_lambda = abs(self.returns_history[-1]) / np.sqrt(volume)
        else:
            kyle_lambda = 0.0
        
        # Volume imbalance ratio
        avg_volume = np.mean(self.volume_history) if len(self.volume_history) > 0 else volume
        volume_imbalance_ratio = volume / (avg_volume + 1e-8)
        
        # Depth imbalance (multi-level if available)
        depth_imbalance = bid_ask_imbalance  # Simplified for now
        
        # Microprice and weighted mid
        if bid_qty + ask_qty > 0:
            micro_price = (bid_qty * current_price * 1.001 + ask_qty * current_price * 0.999) / (bid_qty + ask_qty)
            weighted_mid_price = micro_price
        else:
            micro_price = current_price
            weighted_mid_price = current_price
        
        return MarketMicrostructure(
            bid_qty=bid_qty,
            ask_qty=ask_qty,
            buy_qty=buy_qty,
            sell_qty=sell_qty,
            volume=volume,
            bid_ask_spread=spread,
            bid_ask_imbalance=bid_ask_imbalance,
            order_flow_imbalance=order_flow_imbalance,
            kyle_lambda=kyle_lambda,
            volume_imbalance_ratio=volume_imbalance_ratio,
            depth_imbalance=depth_imbalance,
            micro_price=micro_price,
            weighted_mid_price=weighted_mid_price
        )
    
    def _calculate_market_indicators(self, row: pd.Series) -> List[float]:
        """Calculate broader market indicators"""
        
        indicators = []
        
        # Volume profile
        if len(self.volume_history) > 20:
            volumes = np.array(self.volume_history)
            indicators.extend([
                np.percentile(volumes, 25),
                np.percentile(volumes, 50),
                np.percentile(volumes, 75),
                volumes[-1] / np.mean(volumes),  # Relative volume
                np.std(volumes) / np.mean(volumes)  # Volume volatility
            ])
        else:
            indicators.extend([0] * 5)
        
        # Price levels
        if len(self.price_history) > 20:
            prices = np.array(self.price_history)
            indicators.extend([
                (prices[-1] - np.min(prices)) / (np.max(prices) - np.min(prices) + 1e-8),  # Price position
                prices[-1] / np.mean(prices),  # Relative price
                self._calculate_support_resistance(prices)
            ])
        else:
            indicators.extend([0] * 3)
        
        return indicators
    
    def _extract_temporal_features(self) -> List[float]:
        """Extract time-based features"""
        
        # For crypto markets, we might want to track:
        # - Time since last significant move
        # - Periodic patterns (if timestamp available)
        # - Number of consecutive up/down moves
        
        features = []
        
        if len(self.returns_history) > 1:
            returns = np.array(self.returns_history)
            
            # Consecutive moves
            ups = sum(1 for r in returns[-10:] if r > 0)
            downs = sum(1 for r in returns[-10:] if r < 0)
            features.extend([ups / 10.0, downs / 10.0])
            
            # Time since large move
            large_move_threshold = 2 * np.std(returns) if len(returns) > 10 else 0.01
            time_since_large = 0
            for i in range(len(returns) - 1, -1, -1):
                if abs(returns[i]) > large_move_threshold:
                    break
                time_since_large += 1
            features.append(time_since_large / 100.0)
        else:
            features.extend([0, 0, 0])
        
        return features
    
    def _extract_proprietary_features(self, row: pd.Series) -> List[float]:
        """Extract and process proprietary features X1-X780"""
        
        features = []
        
        # Extract X features
        for i in range(1, 781):
            feature_name = f'X{i}'
            if feature_name in row:
                features.append(float(row[feature_name]))
            else:
                features.append(0.0)
        
        # Apply noise filtering if needed
        if len(features) > 0:
            features = self._filter_features(features)
        
        return features
    
    def _filter_features(self, features: List[float]) -> List[float]:
        """Apply Savitzky-Golay filter for noise reduction"""
        
        if len(features) > 5:
            try:
                # Apply filter with appropriate parameters
                filtered = savgol_filter(features, window_length=min(5, len(features)), polyorder=2)
                return filtered.tolist()
            except:
                return features
        return features
    
    def _calculate_support_resistance(self, prices: np.ndarray) -> float:
        """Calculate distance to nearest support/resistance level"""
        
        if len(prices) < 20:
            return 0.0
        
        # Simple support/resistance using local extrema
        from scipy.signal import argrelextrema
        
        # Find local minima (support) and maxima (resistance)
        local_min = argrelextrema(prices, np.less, order=5)[0]
        local_max = argrelextrema(prices, np.greater, order=5)[0]
        
        current_price = prices[-1]
        
        # Find nearest support and resistance
        if len(local_min) > 0:
            support_levels = prices[local_min]
            nearest_support = support_levels[support_levels < current_price]
            support_dist = (current_price - nearest_support[-1]) / current_price if len(nearest_support) > 0 else 0
        else:
            support_dist = 0
        
        if len(local_max) > 0:
            resistance_levels = prices[local_max]
            nearest_resistance = resistance_levels[resistance_levels > current_price]
            resistance_dist = (nearest_resistance[0] - current_price) / current_price if len(nearest_resistance) > 0 else 0
        else:
            resistance_dist = 0
        
        # Return normalized distance to nearest level
        return min(support_dist, resistance_dist)
    
    def _calculate_reward(self, action: float, actual_return: float, row: pd.Series) -> Tuple[float, Dict]:
        """
        Calculate multi-objective reward with CVaR-based risk penalty
        
        R_t = α1 · Return_t + α2 · Risk_penalty_t + α3 · Transaction_cost_t + α4 · Inventory_penalty_t
        """
        
        # Reward components weights
        alpha1 = 1.0      # Return weight
        alpha2 = -0.5     # Risk penalty weight
        alpha3 = -0.1     # Transaction cost weight
        alpha4 = -0.05    # Inventory penalty weight
        
        components = {}
        
        # 1. Prediction accuracy component (main objective)
        prediction_error = abs(action - actual_return)
        accuracy_reward = -prediction_error  # Negative because we want to minimize error
        components['accuracy'] = accuracy_reward
        
        # 2. Directional accuracy bonus
        direction_correct = (action * actual_return) > 0
        direction_bonus = 0.1 if direction_correct else -0.05
        components['direction'] = direction_bonus
        
        # 3. Risk penalty using CVaR
        risk_penalty = self._calculate_cvar_penalty()
        components['risk_penalty'] = risk_penalty
        
        # 4. Transaction cost
        position_change = abs(action) * self.max_position
        transaction_cost = position_change * self.transaction_cost
        components['transaction_cost'] = -transaction_cost
        
        # 5. Inventory penalty (position limits)
        self.current_position = np.clip(self.current_position + action, -self.max_position, self.max_position)
        inventory_penalty = (abs(self.current_position) / self.max_position) ** 2
        components['inventory_penalty'] = -inventory_penalty
        
        # 6. Sharpe ratio component
        if len(self.returns_history) > 30:
            returns = np.array(self.returns_history[-30:])
            sharpe = (np.mean(returns) - self.risk_free_rate) / (np.std(returns) + 1e-8) * np.sqrt(252 * 24 * 60)
            sharpe_reward = 0.01 * sharpe
            components['sharpe'] = sharpe_reward
        else:
            components['sharpe'] = 0.0
        
        # Combine components
        total_reward = (
            accuracy_reward +
            direction_bonus +
            alpha2 * risk_penalty +
            alpha3 * transaction_cost +
            alpha4 * inventory_penalty +
            components['sharpe']
        )
        
        # Update portfolio value for tracking
        self.portfolio_value *= (1 + actual_return * self.current_position)
        self.peak_value = max(self.peak_value, self.portfolio_value)
        self.drawdown = (self.peak_value - self.portfolio_value) / self.peak_value
        
        return total_reward, components
    
    def _calculate_cvar_penalty(self) -> float:
        """Calculate Conditional Value at Risk (CVaR) penalty"""
        
        if len(self.returns_history) < 100:
            return 0.0
        
        returns
