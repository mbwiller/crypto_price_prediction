import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional, Union
from dataclasses import dataclass
from collections import deque
import warnings
from scipy import stats
from scipy.signal import savgol_filter
import talib
from sklearn.preprocessing import RobustScaler, StandardScaler
from abc import ABC, abstractmethod

warnings.filterwarnings('ignore')

@dataclass
class MarketMicrostructure:
    """Advanced order book and microstructure features"""
    bid_qty: float
    ask_qty: float
    buy_qty: float
    sell_qty: float
    volume: float
    bid_ask_spread: float
    bid_ask_imbalance: float
    order_flow_imbalance: float
    kyle_lambda: float
    volume_imbalance_ratio: float
    depth_imbalance: float
    micro_price: float
    weighted_mid_price: float
