class CryptoPricePredictionMDP:
    """
    Advanced MDP formulation for minute-level cryptocurrency price prediction
    Implements hierarchical state representation with multi-dimensional features
    """
    
    def __init__(self, 
                 lookback_window: int = 100,
                 prediction_horizon: int = 1,
                 use_proprietary_features: bool = True,
                 risk_free_rate: float = 0.02,
                 transaction_cost: float = 0.001,
                 max_position: float = 1.0,
                 cvar_alpha: float = 0.05,
                 risk_tolerance: float = 0.1):
        
        # Environment parameters
        self.lookback_window = lookback_window
        self.prediction_horizon = prediction_horizon
        self.use_proprietary_features = use_proprietary_features
        self.risk_free_rate = risk_free_rate / (365 * 24 * 60)  # Per minute
        self.transaction_cost = transaction_cost
        self.max_position = max_position
        self.cvar_alpha = cvar_alpha
        self.risk_tolerance = risk_tolerance
        
        # Component initialization
        self.tech_indicators = TechnicalIndicators(lookback_window)
        self.vol_regime = VolatilityRegimeDetector()
        
        # State and history tracking
        self.price_history = deque(maxlen=lookback_window)
        self.volume_history = deque(maxlen=lookback_window)
        self.high_history = deque(maxlen=lookback_window)
        self.low_history = deque(maxlen=lookback_window)
        self.returns_history = deque(maxlen=lookback_window)
        self.action_history = deque(maxlen=lookback_window)
        self.reward_history = deque(maxlen=1000)
        
        # Normalization
        self.state_scaler = RobustScaler()
        self.is_fitted = False
        
        # State space definition
        self.state_dim = None  # Will be set after first observation
        self.action_dim = 1    # Continuous price prediction
        
        # Risk metrics
        self.current_position = 0.0
        self.portfolio_value = 1.0
        self.peak_value = 1.0
        self.drawdown = 0.0
        
    def reset(self, initial_data: pd.DataFrame) -> np.ndarray:
        """Reset environment with initial data"""
        
        # Clear histories
        self.price_history.clear()
        self.volume_history.clear()
        self.high_history.clear()
        self.low_history.clear()
        self.returns_history.clear()
        self.action_history.clear()
        
        # Reset risk metrics
        self.current_position = 0.0
        self.portfolio_value = 1.0
        self.peak_value = 1.0
        self.drawdown = 0.0
        
        # Initialize with historical data
        for idx in range(min(len(initial_data), self.lookback_window)):
            row = initial_data.iloc[idx]
            self.price_history.append(row['close'])
            self.volume_history.append(row['volume'])
            self.high_history.append(row.get('high', row['volume']))
            self.low_history.append(row.get('low', row['volume']))
            
            if len(self.price_history) > 1:
                ret = (self.price_history[-1] - self.price_history[-2]) / self.price_history[-2]
                self.returns_history.append(ret)
        
        # Get initial state
        state = self._get_state(initial_data.iloc[-1])
        return state
    
    def step(self, action: float, next_row: pd.Series) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute one step in the environment
        
        Args:
            action: Predicted price change (continuous value)
            next_row: Next observation from dataset
            
        Returns:
            next_state, reward, done, info
        """
        
        # Record action
        self.action_history.append(action)
        
        # Update price history
        current_price = self.price_history[-1]
        next_price = next_row['close']
        actual_return = (next_price - current_price) / current_price
        
        # Calculate reward
        reward, reward_components = self._calculate_reward(action, actual_return, next_row)
        self.reward_history.append(reward)
        
        # Update histories
        self._update_histories(next_row)
        
        # Get next state
        next_state = self._get_state(next_row)
        
        # Check if done (could be based on drawdown, time, etc.)
        done = self._check_done()
        
        # Compile info
        info = {
            'actual_return': actual_return,
            'predicted_return': action,
            'prediction_error': abs(action - actual_return),
            'reward_components': reward_components,
            'portfolio_value': self.portfolio_value,
            'position': self.current_position,
            'drawdown': self.drawdown,
            'volatility_regime': self.vol_regime.detect_regime(np.array(self.returns_history))
        }
        
        return next_state, reward, done, info
    
    def _get_state(self, current_row: pd.Series) -> np.ndarray:
        """Construct comprehensive state representation"""
        
        state_components = []
        
        # 1. Market Microstructure Features
        microstructure = self._calculate_microstructure(current_row)
        state_components.extend([
            microstructure.bid_ask_spread,
            microstructure.bid_ask_imbalance,
            microstructure.order_flow_imbalance,
            microstructure.kyle_lambda,
            microstructure.volume_imbalance_ratio,
            microstructure.depth_imbalance,
            microstructure.micro_price,
            microstructure.weighted_mid_price
        ])

        
# ==========================================================
# GO THROUGH THESE BELOW!!!!!
# ==========================================================
        # 2. Price Information & Returns
        if len(self.price_history) > 1:
            returns = np.array(self.returns_history)
        
        # Calculate VaR at alpha level
        var_alpha = np.percentile(returns, self.cvar_alpha * 100)
        
        # Calculate CVaR (expected value of returns below VaR)
        cvar = np.mean(returns[returns <= var_alpha])

        if abs(cvar) > self.risk_tolerance:
            penalty = (abs(cvar) - self.risk_tolerance) * 10
        else:
            penalty = 0.0
        state_components.append(penalty)

        # 3. [Optional] Technical indicators
        tech_feats = self.tech_indicators.calculate_indicators(
            self.volume_history,
            np.array(self.high_history),
            np.array(self.low_history),
            np.array(self.price_history)      # using price as close
        )
        state_components.extend(tech_feats.values())

        # 4. Volatility regime
        vol_reg = self.vol_regime.detect_regime(np.array(self.returns_history))
        state_components.extend(vol_reg.values())

        # 5. [Placeholder for MarketRegimeClassifier & other indicators…]
        #    state_components.extend(market_regime_feats)

        # 6. Proprietary features if enabled
        if self.use_proprietary_features:
            prop = [current_row[f'X_{i}'] for i in range(1,781)]
            state_components.extend(prop)

        # 7. Final assembly
        state = np.array(state_components, dtype=float)

        # 8. Record the dimension once
        if self.state_dim is None:
            self.state_dim = state.shape[0]

        # 9. [Optional] Scale
        # if not self.is_fitted:
        #     self.state_scaler.fit(state.reshape(1,-1))
        #     self.is_fitted = True
        # state = self.state_scaler.transform(state.reshape(1,-1))[0]

        return state
    
    def _update_histories(self, row: pd.Series):
        """Update all history buffers"""
        
        self.price_history.append(row['close'])
        self.volume_history.append(row['volume'])
        self.high_history.append(row.get('high', row['volume']))
        self.low_history.append(row.get('low', row['volume']))
        
        if len(self.price_history) > 1:
            ret = (self.price_history[-1] - self.price_history[-2]) / self.price_history[-2]
            self.returns_history.append(ret)
    
    def _check_done(self) -> bool:
        """Check if episode should terminate"""
        
        # Terminate on excessive drawdown
        if self.drawdown > 0.2:  # 20% drawdown
            return True
        
        # Terminate on position limits breach
        if abs(self.current_position) > self.max_position * 1.1:
            return True
        
        return False
    
    def get_action_space_info(self) -> Dict:
        """Get information about the action space"""
        return {
            'type': 'continuous',
            'shape': (self.action_dim,),
            'low': -0.1,  # Maximum 10% price change prediction
            'high': 0.1,
            'description': 'Predicted price change (return) for next minute'
        }
    
    def get_state_space_info(self) -> Dict:
        """Get information about the state space"""
        if self.state_dim is None:
            return {'type': 'continuous', 'shape': 'Variable - depends on features'}
        
        return {
            'type': 'continuous',
            'shape': (self.state_dim,),
            'components': {
                'microstructure': 8,
                'price_returns': 7,
                'technical_indicators': 40,
                'volatility_regime': 7,
                'market_indicators': 8,
                'temporal': 3,
                'portfolio': 4,
                'proprietary': 780 if self.use_proprietary_features else 0
            },
            'total_dim': self.state_dim
        }
