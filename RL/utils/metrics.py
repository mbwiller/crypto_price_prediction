import numpy as np
from typing import Dict, List, Tuple
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd

def calculate_metrics(predictions: np.ndarray, actuals: np.ndarray) -> Dict[str, float]:
    """
    Calculate comprehensive metrics for predictions
    """
    
    # Basic metrics
    mse = mean_squared_error(actuals, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actuals, predictions)
    
    # R-squared
    r2 = r2_score(actuals, predictions)
    
    # Directional accuracy
    pred_direction = np.sign(predictions)
    actual_direction = np.sign(actuals)
    directional_accuracy = np.mean(pred_direction == actual_direction)
    
    # Correlation
    correlation = np.corrcoef(predictions, actuals)[0, 1]
    
    # Quantile metrics
    errors = np.abs(predictions - actuals)
    q50 = np.median(errors)
    q90 = np.percentile(errors, 90)
    q95 = np.percentile(errors, 95)
    
    # Information ratio (if returns)
    if len(predictions) > 1:
        tracking_error = np.std(predictions - actuals)
        mean_excess = np.mean(predictions - actuals)
        information_ratio = mean_excess / tracking_error if tracking_error > 0 else 0
    else:
        information_ratio = 0
    
    return {
        'mse': float(mse),
        'rmse': float(rmse),
        'mae': float(mae),
        'r2': float(r2),
        'directional_accuracy': float(directional_accuracy),
        'correlation': float(correlation),
        'median_error': float(q50),
        'q90_error': float(q90),
        'q95_error': float(q95),
        'information_ratio': float(information_ratio)
    }

def calculate_trading_metrics(returns: np.ndarray, predictions: np.ndarray) -> Dict[str, float]:
    """
    Calculate trading-specific metrics
    """
    
    # Sharpe ratio (annualized)
    sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252 * 24 * 60) if np.std(returns) > 0 else 0
    
    # Maximum drawdown
    cumulative = (1 + returns).cumprod()
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = np.min(drawdown)
    
    # Hit rate
    hit_rate = np.mean((returns > 0) == (predictions > 0))
    
    return {
        'sharpe_ratio': float(sharpe),
        'max_drawdown': float(max_drawdown),
        'hit_rate': float(hit_rate)
    }
