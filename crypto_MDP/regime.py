class VolatilityRegimeDetector:
    """Detect and classify market volatility regimes"""
    
    def __init__(self, window_short: int = 10, window_long: int = 60):
        self.window_short = window_short
        self.window_long = window_long
        self.regime_history = deque(maxlen=1000)
        
    def detect_regime(self, returns: np.ndarray) -> Dict[str, float]:
        """Detect current volatility regime using multiple methods"""
        
        # Calculate rolling volatilities
        if len(returns) < self.window_long:
            return self._default_regime()
            
        vol_short = np.std(returns[-self.window_short:]) * np.sqrt(252 * 24 * 60)  # Annualized
        vol_long = np.std(returns[-self.window_long:]) * np.sqrt(252 * 24 * 60)
        
        # GARCH-style conditional volatility
        ewma_vol = self._calculate_ewma_volatility(returns)
        
        # Regime classification
        vol_ratio = vol_short / (vol_long + 1e-8)
        
        regime_features = {
            'vol_short': vol_short,
            'vol_long': vol_long,
            'vol_ratio': vol_ratio,
            'ewma_vol': ewma_vol,
            'is_high_vol': float(vol_short > np.percentile(self.regime_history, 75) if len(self.regime_history) > 100 else vol_short > 0.5),
            'is_trending': float(abs(np.mean(returns[-self.window_short:])) > 2 * vol_short / np.sqrt(self.window_short)),
            'vol_of_vol': np.std([np.std(returns[i:i+5]) for i in range(len(returns)-5)]) if len(returns) > 10 else 0
        }
        
        self.regime_history.append(vol_short)
        return regime_features
    
    def _calculate_ewma_volatility(self, returns: np.ndarray, lambda_param: float = 0.94) -> float:
        """Calculate EWMA volatility"""
        if len(returns) < 2:
            return 0.0
        
        weights = np.array([(1 - lambda_param) * lambda_param ** i for i in range(len(returns))])
        weights = weights / weights.sum()
        weighted_returns = returns ** 2
        ewma_var = np.sum(weights * weighted_returns)
        return np.sqrt(ewma_var) * np.sqrt(252 * 24 * 60)
    
    def _default_regime(self) -> Dict[str, float]:
        """Return default regime when insufficient data"""
        return {
            'vol_short': 0.3,
            'vol_long': 0.3,
            'vol_ratio': 1.0,
            'ewma_vol': 0.3,
            'is_high_vol': 0.0,
            'is_trending': 0.0,
            'vol_of_vol': 0.0
        }
