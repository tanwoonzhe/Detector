"""
技术指标计算模块
================================
使用ta库计算各类技术指标: RSI, MACD, 布林带, 移动平均线等
参考: John Murphy《Technical Analysis of the Financial Markets》
"""

import pandas as pd
import numpy as np
import ta
from ta.trend import SMAIndicator, EMAIndicator, MACD, ADXIndicator
from ta.momentum import RSIIndicator, StochasticOscillator, ROCIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator, VolumeWeightedAveragePrice
import logging

from config import FeatureConfig
from typing import Optional

logger = logging.getLogger(__name__)


class TechnicalIndicators:
    """
    技术指标计算类
    
    包含:
    - 趋势指标: SMA, EMA, MACD, ADX
    - 动量指标: RSI, Stochastic, ROC
    - 波动率指标: Bollinger Bands, ATR
    - 成交量指标: OBV, VWAP
    """
    
    def __init__(self, config: Optional[FeatureConfig] = None):
        self.config = config or FeatureConfig()
    
    def add_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        添加所有技术指标
        
        Args:
            df: 必须包含 open, high, low, close, volume 列
            
        Returns:
            添加指标后的DataFrame
        """
        df = df.copy()
        
        # 验证必要列
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"缺少必要列: {col}")
        
        # 添加各类指标
        df = self.add_trend_indicators(df)
        df = self.add_momentum_indicators(df)
        df = self.add_volatility_indicators(df)
        df = self.add_volume_indicators(df)
        df = self.add_return_features(df)
        df = self.add_time_features(df)
        
        logger.info(f"添加技术指标完成，新增 {len(df.columns) - len(required_cols)} 个特征")
        return df
    
    def add_trend_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加趋势指标"""
        close = df['close']
        
        # 简单移动平均线 (SMA)
        for period in self.config.SMA_PERIODS:
            sma = SMAIndicator(close, window=period)
            df[f'sma_{period}'] = sma.sma_indicator()
            # 价格相对于SMA的位置
            df[f'price_sma_{period}_ratio'] = close / df[f'sma_{period}']
        
        # 指数移动平均线 (EMA)
        for period in self.config.EMA_PERIODS:
            ema = EMAIndicator(close, window=period)
            df[f'ema_{period}'] = ema.ema_indicator()
        
        # MACD
        macd = MACD(
            close,
            window_slow=self.config.MACD_SLOW,
            window_fast=self.config.MACD_FAST,
            window_sign=self.config.MACD_SIGNAL
        )
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_histogram'] = macd.macd_diff()
        
        # MACD交叉信号
        df['macd_cross'] = np.where(
            df['macd'] > df['macd_signal'], 1,
            np.where(df['macd'] < df['macd_signal'], -1, 0)
        )
        
        # ADX (Average Directional Index) - 趋势强度
        # 只在数据量足够时计算
        if len(df) >= 14:
            adx = ADXIndicator(df['high'], df['low'], close, window=14)
            df['adx'] = adx.adx()
            df['adx_pos'] = adx.adx_pos()
            df['adx_neg'] = adx.adx_neg()
        else:
            df['adx'] = 0
            df['adx_pos'] = 0
            df['adx_neg'] = 0
        
        # 趋势方向
        df['trend_direction'] = np.where(
            df['adx_pos'] > df['adx_neg'], 1,
            np.where(df['adx_pos'] < df['adx_neg'], -1, 0)
        )
        
        return df
    
    def add_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加动量指标"""
        close = df['close']
        high = df['high']
        low = df['low']
        
        # RSI (Relative Strength Index)
        rsi = RSIIndicator(close, window=self.config.RSI_PERIOD)
        df['rsi'] = rsi.rsi()
        
        # RSI超买超卖信号
        df['rsi_overbought'] = (df['rsi'] > 70).astype(int)
        df['rsi_oversold'] = (df['rsi'] < 30).astype(int)
        
        # Stochastic Oscillator
        stoch = StochasticOscillator(high, low, close, window=14, smooth_window=3)
        df['stoch_k'] = stoch.stoch()
        df['stoch_d'] = stoch.stoch_signal()
        
        # ROC (Rate of Change)
        for period in [1, 6, 12, 24]:
            roc = ROCIndicator(close, window=period)
            df[f'roc_{period}'] = roc.roc()
        
        # 动量 (当前价格与N期前的差)
        for period in [1, 4, 12, 24]:
            df[f'momentum_{period}'] = close - close.shift(period)
        
        return df
    
    def add_volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加波动率指标"""
        close = df['close']
        high = df['high']
        low = df['low']
        
        # Bollinger Bands
        bb = BollingerBands(
            close,
            window=self.config.BB_PERIOD,
            window_dev=self.config.BB_STD
        )
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_middle'] = bb.bollinger_mavg()
        df['bb_lower'] = bb.bollinger_lband()
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        
        # Bollinger Band位置 (0-1范围)
        df['bb_position'] = (close - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # ATR (Average True Range)
        atr = AverageTrueRange(high, low, close, window=14)
        df['atr'] = atr.average_true_range()
        
        # ATR比率 (波动率相对于价格)
        df['atr_ratio'] = df['atr'] / close
        
        # 历史波动率
        for period in [6, 12, 24]:
            df[f'volatility_{period}'] = close.pct_change().rolling(window=period).std()
        
        # 日内波动
        df['intraday_range'] = (high - low) / close
        
        return df
    
    def add_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加成交量指标"""
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']
        
        # OBV (On-Balance Volume)
        obv = OnBalanceVolumeIndicator(close, volume)
        df['obv'] = obv.on_balance_volume()
        
        # OBV变化率
        df['obv_change'] = df['obv'].pct_change()
        
        # VWAP (Volume Weighted Average Price)
        # 使用滚动窗口计算避免累积和过大
        typical_price = (high + low + close) / 3
        vwap_window = 24  # 24小时窗口
        vwap_numerator = (typical_price * volume).rolling(window=vwap_window, min_periods=1).sum()
        vwap_denominator = volume.rolling(window=vwap_window, min_periods=1).sum()
        df['vwap'] = vwap_numerator / vwap_denominator.replace(0, np.nan)
        
        # 价格相对VWAP位置（避免除以0）
        df['price_vwap_ratio'] = close / df['vwap'].replace(0, np.nan)
        
        # 成交量变化
        for period in [1, 6, 12, 24]:
            df[f'volume_change_{period}'] = volume.pct_change(period)
            df[f'volume_sma_{period}'] = volume.rolling(window=period, min_periods=1).mean()
        
        # 成交量比率 (当前成交量/平均成交量)，避免除以0
        volume_ma24 = volume.rolling(window=24, min_periods=1).mean()
        df['volume_ratio'] = volume / volume_ma24.replace(0, np.nan)
        
        return df
    
    def add_return_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加收益率特征"""
        close = df['close']
        
        # 对数收益率
        df['log_return'] = np.log(close / close.shift(1))
        
        # 多周期收益率
        for period in self.config.RETURN_PERIODS:
            df[f'return_{period}h'] = close.pct_change(period)
            df[f'log_return_{period}h'] = np.log(close / close.shift(period))
        
        # 累计收益率
        for period in [12, 24, 48]:
            df[f'cum_return_{period}h'] = close.pct_change(period)
        
        return df
    
    def add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加时间特征"""
        if not isinstance(df.index, pd.DatetimeIndex):
            # 尝试从timestamp列创建
            if 'timestamp' in df.columns:
                df.set_index('timestamp', inplace=True)
            else:
                logger.warning("无法添加时间特征: 缺少时间索引")
                return df
        
        # 小时特征 - 使用DatetimeIndex属性
        dt_index = pd.DatetimeIndex(df.index)
        df['hour'] = dt_index.hour
        df['day_of_week'] = dt_index.dayofweek
        df['day_of_month'] = dt_index.day
        df['is_weekend'] = (dt_index.dayofweek >= 5).astype(int)
        
        # 交易时段特征
        # 亚洲时段: 00:00-08:00 UTC
        # 欧洲时段: 08:00-16:00 UTC
        # 美洲时段: 16:00-24:00 UTC
        df['is_asian_session'] = ((df['hour'] >= 0) & (df['hour'] < 8)).astype(int)
        df['is_european_session'] = ((df['hour'] >= 8) & (df['hour'] < 16)).astype(int)
        df['is_american_session'] = ((df['hour'] >= 16) & (df['hour'] < 24)).astype(int)
        
        # 周期性编码 (正弦/余弦)
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        return df
    
    def get_feature_names(self) -> list:
        """获取所有特征名称"""
        return [
            # 趋势
            *[f'sma_{p}' for p in self.config.SMA_PERIODS],
            *[f'price_sma_{p}_ratio' for p in self.config.SMA_PERIODS],
            *[f'ema_{p}' for p in self.config.EMA_PERIODS],
            'macd', 'macd_signal', 'macd_histogram', 'macd_cross',
            'adx', 'adx_pos', 'adx_neg', 'trend_direction',
            # 动量
            'rsi', 'rsi_overbought', 'rsi_oversold',
            'stoch_k', 'stoch_d',
            *[f'roc_{p}' for p in [1, 6, 12, 24]],
            *[f'momentum_{p}' for p in [1, 4, 12, 24]],
            # 波动率
            'bb_upper', 'bb_middle', 'bb_lower', 'bb_width', 'bb_position',
            'atr', 'atr_ratio',
            *[f'volatility_{p}' for p in [6, 12, 24]],
            'intraday_range',
            # 成交量
            'obv', 'obv_change', 'vwap', 'price_vwap_ratio',
            *[f'volume_change_{p}' for p in [1, 6, 12, 24]],
            *[f'volume_sma_{p}' for p in [1, 6, 12, 24]],
            'volume_ratio',
            # 收益率
            'log_return',
            *[f'return_{p}h' for p in self.config.RETURN_PERIODS],
            *[f'log_return_{p}h' for p in self.config.RETURN_PERIODS],
            *[f'cum_return_{p}h' for p in [12, 24, 48]],
            # 时间
            'hour', 'day_of_week', 'day_of_month', 'is_weekend',
            'is_asian_session', 'is_european_session', 'is_american_session',
            'hour_sin', 'hour_cos', 'day_sin', 'day_cos'
        ]


# 全局实例
technical_indicators = TechnicalIndicators()
