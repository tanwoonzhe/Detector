"""
特征工程主模块
================================
整合所有特征: 技术指标、蜡烛图形态、支撑阻力、情感特征
生成训练用的标准化特征矩阵
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import Tuple, Optional, List, Dict, Any, Union
import logging

from .technical import TechnicalIndicators, technical_indicators
from .patterns import CandlestickPatterns, candlestick_patterns
from .support_resistance import SupportResistance, support_resistance
from config import ModelConfig, TradingConfig, FeatureConfig

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    特征工程主类
    
    功能:
    1. 整合所有特征计算模块
    2. 合并情感数据
    3. 创建预测标签 (方向 + 幅度)
    4. 特征标准化
    5. 生成序列数据 (用于RNN)
    """
    
    def __init__(
        self,
        sequence_length: Optional[int] = None,
        prediction_windows: Optional[List[float]] = None,
        sideways_threshold: Optional[float] = None
    ):
        self.sequence_length = sequence_length or ModelConfig.SEQUENCE_LENGTH
        self.prediction_windows = prediction_windows or TradingConfig.PREDICTION_WINDOWS
        self.sideways_threshold = sideways_threshold or TradingConfig.SIDEWAYS_THRESHOLD
        
        # 特征计算模块
        self.technical = technical_indicators
        self.patterns = candlestick_patterns
        self.support_resistance = support_resistance
        
        # 标准化器
        self.feature_scaler = StandardScaler()
        self.target_scaler = MinMaxScaler(feature_range=(-1, 1))
        
        self._feature_columns: List[str] = []
        self._is_fitted = False
    
    def create_features(
        self, 
        ohlcv_df: pd.DataFrame,
        sentiment_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        创建所有特征
        
        Args:
            ohlcv_df: OHLCV数据，必须包含 timestamp, open, high, low, close, volume
            sentiment_df: 情感数据，包含 timestamp, composite, fear_greed, news, reddit
            
        Returns:
            包含所有特征的DataFrame
        """
        logger.info("开始特征工程...")
        
        # 确保索引是时间戳
        if 'timestamp' in ohlcv_df.columns:
            ohlcv_df = ohlcv_df.set_index('timestamp')
        
        df = ohlcv_df.copy()
        
        # 添加技术指标
        logger.info("  计算技术指标...")
        df = self.technical.add_all_indicators(df)
        
        # 添加蜡烛图形态
        logger.info("  检测蜡烛图形态...")
        df = self.patterns.add_all_patterns(df)
        
        # 添加支撑阻力位
        logger.info("  计算支撑阻力位...")
        df = self.support_resistance.add_all_features(df)
        
        # 合并情感数据
        if sentiment_df is not None and not sentiment_df.empty:
            logger.info("  合并情感数据...")
            df = self._merge_sentiment(df, sentiment_df)
        else:
            # 添加占位符情感特征
            df['sentiment_composite'] = 0
            df['sentiment_fear_greed'] = 0
            df['sentiment_news'] = 0
            df['sentiment_reddit'] = 0
            df['sentiment_momentum'] = 0
        
        # 移除包含NaN的行 (技术指标需要历史数据)
        initial_len = len(df)
        df = df.dropna()
        logger.info(f"  移除 {initial_len - len(df)} 行含NaN的数据")
        
        return df
    
    def _merge_sentiment(
        self, 
        df: pd.DataFrame, 
        sentiment_df: pd.DataFrame
    ) -> pd.DataFrame:
        """合并情感数据"""
        # 确保情感数据索引是时间戳
        if 'timestamp' in sentiment_df.columns:
            sentiment_df = sentiment_df.set_index('timestamp')
        
        # 重命名列避免冲突
        sentiment_cols = {}
        for col in sentiment_df.columns:
            if col not in ['timestamp']:
                sentiment_cols[col] = f'sentiment_{col}'
        
        sentiment_df = sentiment_df.rename(columns=sentiment_cols)
        
        # 按时间戳合并 (最近邻匹配)
        df = df.join(sentiment_df, how='left')
        
        # 前向填充缺失的情感数据 (Fear & Greed是日级别)
        sentiment_columns = [c for c in df.columns if c.startswith('sentiment_')]
        df[sentiment_columns] = df[sentiment_columns].ffill().fillna(0)
        
        # 添加情感动量 (4小时变化)
        if 'sentiment_composite' in df.columns:
            df['sentiment_momentum'] = df['sentiment_composite'] - df['sentiment_composite'].shift(4)
        
        return df
    
    def create_labels(
        self, 
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        创建预测标签
        
        为每个预测窗口创建:
        - direction: 方向 (0=下跌, 1=横盘, 2=上涨)
        - magnitude: 幅度 (收益率)
        """
        close = df['close']
        
        for window in self.prediction_windows:
            # 窗口对应的小时数
            periods = int(window) if window >= 1 else 1
            
            # 未来收益率
            future_return = close.shift(-periods) / close - 1
            
            col_prefix = f'target_{window}h'
            
            # 收益率 (回归目标)
            df[f'{col_prefix}_return'] = future_return
            
            # 方向 (分类目标)
            df[f'{col_prefix}_direction'] = np.where(
                future_return > self.sideways_threshold, 2,  # 上涨
                np.where(future_return < -self.sideways_threshold, 0, 1)  # 下跌/横盘
            )
        
        # 移除最后几行 (没有未来数据)
        max_periods = int(max(self.prediction_windows))
        if max_periods > 0:
            df = pd.DataFrame(df.iloc[:-max_periods])
        
        return df
    
    def get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """获取特征列名"""
        # 排除原始OHLCV和目标列
        exclude_prefixes = ['open', 'high', 'low', 'close', 'volume', 'target_']
        
        feature_cols = [
            col for col in df.columns 
            if not any(col.startswith(p) or col == p for p in exclude_prefixes)
        ]
        
        return feature_cols
    
    def prepare_training_data(
        self, 
        df: pd.DataFrame,
        target_window: float = 1.0,
        for_classification: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        准备训练数据
        
        Args:
            df: 包含特征和标签的DataFrame
            target_window: 目标预测窗口
            for_classification: True返回分类标签, False返回回归标签
            
        Returns:
            (X, y, feature_names)
        """
        feature_cols = self.get_feature_columns(df)
        self._feature_columns = feature_cols
        
        X = df[feature_cols].values
        
        if for_classification:
            y = df[f'target_{target_window}h_direction'].values
        else:
            y = df[f'target_{target_window}h_return'].values
        
        # 标准化特征
        if not self._is_fitted:
            X = self.feature_scaler.fit_transform(X)
            self._is_fitted = True
        else:
            X = self.feature_scaler.transform(X)
        
        return X, y, feature_cols
    
    def create_sequences(
        self, 
        X: np.ndarray, 
        y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        创建序列数据 (用于RNN/LSTM/GRU)
        
        Args:
            X: 特征矩阵 (samples, features)
            y: 标签数组 (samples,)
            
        Returns:
            (X_seq, y_seq) 
            X_seq: (samples - seq_len, seq_len, features)
            y_seq: (samples - seq_len,)
        """
        seq_len = self.sequence_length
        n_samples = len(X) - seq_len
        n_features = X.shape[1]
        
        X_seq = np.zeros((n_samples, seq_len, n_features))
        y_seq = np.zeros(n_samples)
        
        for i in range(n_samples):
            X_seq[i] = X[i:i+seq_len]
            y_seq[i] = y[i+seq_len]
        
        return X_seq, y_seq
    
    def prepare_multi_target_data(
        self, 
        df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        准备多目标训练数据 (同时预测多个时间窗口)
        
        Returns:
            {
                'X_seq': 序列特征,
                'y_direction': {window: 方向标签},
                'y_return': {window: 收益率标签}
            }
        """
        feature_cols = self.get_feature_columns(df)
        self._feature_columns = feature_cols
        
        X = df[feature_cols].values
        
        # 标准化
        if not self._is_fitted:
            X = self.feature_scaler.fit_transform(X)
            self._is_fitted = True
        else:
            X = self.feature_scaler.transform(X)
        
        # 获取所有目标
        y_direction = {}
        y_return = {}
        
        for window in self.prediction_windows:
            y_direction[window] = df[f'target_{window}h_direction'].values
            y_return[window] = df[f'target_{window}h_return'].values
        
        # 创建序列
        seq_len = self.sequence_length
        n_samples = len(X) - seq_len
        n_features = X.shape[1]
        
        X_seq = np.zeros((n_samples, seq_len, n_features))
        
        for i in range(n_samples):
            X_seq[i] = X[i:i+seq_len]
        
        # 对齐标签
        for window in self.prediction_windows:
            y_direction[window] = y_direction[window][seq_len:]
            y_return[window] = y_return[window][seq_len:]
        
        return {
            'X_seq': X_seq,
            'y_direction': y_direction,
            'y_return': y_return,
            'feature_names': feature_cols
        }
    
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        转换新数据 (预测时使用)
        """
        if not self._is_fitted:
            raise ValueError("FeatureEngineer未拟合，请先调用prepare_training_data")
        
        X = df[self._feature_columns].values
        X = self.feature_scaler.transform(X)
        return X
    
    def get_latest_sequence(self, df: pd.DataFrame) -> np.ndarray:
        """
        获取最新的序列数据 (用于实时预测)
        
        Returns:
            (1, seq_len, features) 的数组
        """
        X = self.transform(df)
        
        if len(X) < self.sequence_length:
            raise ValueError(f"数据不足，需要至少 {self.sequence_length} 条记录")
        
        return X[-self.sequence_length:].reshape(1, self.sequence_length, -1)


# 全局实例
feature_engineer = FeatureEngineer()
