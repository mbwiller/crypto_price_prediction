import numpy as np
import pandas as pd
from typing import Dict
from collections import deque

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

        # CRYPTO TRADES 24/7!!
        minutes_per_year = 365 * 24 * 60
        vol_short = np.std(returns[-self.window_short:]) * np.sqrt(minutes_per_year)
        vol_long  = np.std(returns[-self.window_long:])  * np.sqrt(minutes_per_year)
        
        # GARCH-style conditional volatility
        ewma_vol = self._calculate_ewma_volatility(returns)
        
        # Regime classification
        vol_ratio = vol_short / (vol_long + 1e-8)
        
        regime_features = {
            'vol_short': vol_short,
            'vol_long': vol_long,
            'vol_ratio': vol_ratio,
            'ewma_vol': ewma_vol,
            'is_high_vol': float(len(self.regime_history) > 100 and vol_short > np.percentile(self.regime_history, 75)),
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
        minutes_per_year = 365 * 24 * 60
        return np.sqrt(ewma_var) * np.sqrt(minutes_per_year)
    
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


class MarketRegimeClassifier:
    """
    Classify market into different regimes for adaptive strategies
    """
    
    def __init__(self, n_regimes: int = 4):
        self.n_regimes = n_regimes
        self.regime_centers = None
        
    def fit(self, returns: np.ndarray, volatilities: np.ndarray):
        """
        Fit regime classifier using returns and volatility data
        """
        from sklearn.cluster import KMeans
        
        # Ensure returns is 2D: each row=window of returns
        R = np.atleast_2d(returns)
        # Prepare features per window
        means = R.mean(axis=1)
        skews = stats.skew(R, axis=1)
        kurts = stats.kurtosis(R, axis=1)
        features = np.column_stack([means, volatilities, skews, kurts])
        
        # Cluster into regimes
        kmeans = KMeans(n_clusters=self.n_regimes, random_state=42)
        kmeans.fit(features)
        
        self.regime_centers = kmeans.cluster_centers_
        self.kmeans = kmeans
        
        return self
    
    def predict_regime(self, recent_returns: np.ndarray, recent_volatility: float) -> int:
        """Predict current market regime"""
        
        if self.kmeans is None:
            return 0
        
        features = np.array([
            np.mean(recent_returns),
            recent_volatility,
            stats.skew(recent_returns) if len(recent_returns) > 3 else 0,
            stats.kurtosis(recent_returns) if len(recent_returns) > 3 else 0
        ]).reshape(1, -1)
        
        regime = self.kmeans.predict(features)[0]
        return regime
    
    def get_regime_characteristics(self, regime: int) -> Dict:
        """Get characteristics of a specific regime"""
        
        if self.regime_centers is None:
            return {}
        
        center = self.regime_centers[regime]
        return {
            'mean_return': center[0],
            'volatility': center[1],
            'skewness': center[2],
            'kurtosis': center[3],
            'regime_type': self._classify_regime_type(center)
        }
    
    def _classify_regime_type(self, center: np.ndarray) -> str:
        """Classify regime into human-readable type"""
        
        mean_return, volatility, skewness, kurtosis = center
        
# ==================================================================
        # WE SHOULD LOOK INTO THESE, AS THEY ARE HARD CODED!!
# ==================================================================
        if mean_return > 0.001 and volatility < 0.02:
            return "Bull Market - Low Volatility"
        elif mean_return > 0.001 and volatility >= 0.02:
            return "Bull Market - High Volatility"
        elif mean_return < -0.001 and volatility < 0.02:
            return "Bear Market - Low Volatility"
        elif mean_return < -0.001 and volatility >= 0.02:
            return "Bear Market - High Volatility"
        else:
            return "Sideways Market"
