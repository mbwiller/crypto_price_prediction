import numpy as np
import pandas as pd
import talib
from typing import Dict

class TechnicalIndicators:
    """Advanced technical indicators optimized for crypto markets"""
    
    def __init__(self, lookback_window: int = 100):
        self.lookback_window = lookback_window
        
    def calculate_indicators(self, volumes: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive technical indicators"""
        
        indicators = {}
        
        # Price-based indicators
        indicators['rsi_14'] = self._safe_calc(talib.RSI, close, timeperiod=14)
        indicators['rsi_6'] = self._safe_calc(talib.RSI, close, timeperiod=6)
        
        # Crypto-optimized MACD (8,21,5)
        macd, signal, hist = talib.MACD(close, fastperiod=8, slowperiod=21, signalperiod=5)
        indicators['macd'] = self._safe_value(macd)
        indicators['macd_signal'] = self._safe_value(signal)
        indicators['macd_hist'] = self._safe_value(hist)
        
        # Bollinger Bands with crypto-specific parameters
        upper, middle, lower = talib.BBANDS(close, timeperiod=20, nbdevup=1.5, nbdevdn=1.5)
        indicators['bb_upper'] = self._safe_value(upper)
        indicators['bb_middle'] = self._safe_value(middle)
        indicators['bb_lower'] = self._safe_value(lower)
        indicators['bb_width'] = (indicators['bb_upper'] - indicators['bb_lower']) / indicators['bb_middle']
        indicators['bb_position'] = (close[-1] - indicators['bb_lower']) / (indicators['bb_upper'] - indicators['bb_lower'])
        
        # Volatility indicators
        indicators['atr_14'] = self._safe_calc(talib.ATR, high, low, close, timeperiod=14)
        indicators['atr_7'] = self._safe_calc(talib.ATR, high, low, close, timeperiod=7)
        indicators['natr_14'] = self._safe_calc(talib.NATR, high, low, close, timeperiod=14)
        
        # Volume indicators
        indicators['obv'] = self._safe_calc(talib.OBV, close, volumes)
        indicators['adosc'] = self._safe_calc(talib.ADOSC, high, low, close, volumes, fastperiod=3, slowperiod=10)
        indicators['mfi'] = self._safe_calc(talib.MFI, high, low, close, volumes, timeperiod=14)
        
        # Momentum indicators
        indicators['mom_10'] = self._safe_calc(talib.MOM, close, timeperiod=10)
        indicators['mom_5'] = self._safe_calc(talib.MOM, close, timeperiod=5)
        indicators['roc_10'] = self._safe_calc(talib.ROC, close, timeperiod=10)
        indicators['willr'] = self._safe_calc(talib.WILLR, high, low, close, timeperiod=14)
        indicators['cci'] = self._safe_calc(talib.CCI, high, low, close, timeperiod=14)
        
        # Pattern recognition
        indicators['adx'] = self._safe_calc(talib.ADX, high, low, close, timeperiod=14)
        indicators['plus_di'] = self._safe_calc(talib.PLUS_DI, high, low, close, timeperiod=14)
        indicators['minus_di'] = self._safe_calc(talib.MINUS_DI, high, low, close, timeperiod=14)
        
        # Moving averages
        indicators['sma_5'] = self._safe_calc(talib.SMA, close, timeperiod=5)
        indicators['sma_20'] = self._safe_calc(talib.SMA, close, timeperiod=20)
        indicators['ema_9'] = self._safe_calc(talib.EMA, close, timeperiod=9)
        indicators['ema_21'] = self._safe_calc(talib.EMA, close, timeperiod=21)
        indicators['wma_10'] = self._safe_calc(talib.WMA, close, timeperiod=10)
        
        # Advanced indicators
        indicators['sar'] = self._safe_calc(talib.SAR, high, low)
        indicators['trix'] = self._safe_calc(talib.TRIX, close, timeperiod=14)
        indicators['ultosc'] = self._safe_calc(talib.ULTOSC, high, low, close)
        
        # Stochastic oscillators
        slowk, slowd = talib.STOCH(high, low, close, fastk_period=14, slowk_period=3, slowd_period=3)
        indicators['stoch_k'] = self._safe_value(slowk)
        indicators['stoch_d'] = self._safe_value(slowd)
        
        # Aroon indicators
        aroon_up, aroon_down = talib.AROON(high, low, timeperiod=14)
        indicators['aroon_up'] = self._safe_value(aroon_up)
        indicators['aroon_down'] = self._safe_value(aroon_down)
        indicators['aroon_osc'] = self._safe_calc(talib.AROONOSC, high, low, timeperiod=14)
        
        return indicators
    
    def _safe_calc(self, func, *args, **kwargs):
        """Safely calculate indicator with error handling"""
        try:
            result = func(*args, **kwargs)
            return self._safe_value(result)
        except:
            return 0.0
    
    def _safe_value(self, value):
        """Safely extract last value from array"""
        if isinstance(value, (np.ndarray, pd.Series)):
            if len(value) > 0 and not np.isnan(value[-1]):
                return float(value[-1])
        elif value is not None and not np.isnan(value):
            return float(value)
        return 0.0
