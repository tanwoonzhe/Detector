"""
支撑阻力位检测模块
================================
使用Pivot Points和局部极值检测支撑阻力位
"""

import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
from sklearn.cluster import KMeans
import logging

logger = logging.getLogger(__name__)


class SupportResistance:
    """
    支撑阻力位检测类
    
    方法:
    1. Pivot Points (日内交易常用)
    2. 局部极值检测
    3. K-means聚类 (历史价格聚类)
    """
    
    def __init__(self, window: int = 10, n_clusters: int = 5):
        """
        Args:
            window: 局部极值检测窗口
            n_clusters: 聚类数量
        """
        self.window = window
        self.n_clusters = n_clusters
    
    def add_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加所有支撑阻力特征"""
        df = df.copy()
        
        df = self.add_pivot_points(df)
        df = self.add_local_extrema(df)
        df = self.add_price_levels(df)
        df = self.add_distance_features(df)
        
        logger.info("支撑阻力位检测完成")
        return df
    
    def add_pivot_points(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算Pivot Points
        
        标准Pivot Points公式:
        PP = (H + L + C) / 3
        R1 = 2 * PP - L
        S1 = 2 * PP - H
        R2 = PP + (H - L)
        S2 = PP - (H - L)
        R3 = H + 2 * (PP - L)
        S3 = L - 2 * (H - PP)
        """
        high = df['high']
        low = df['low']
        close = df['close']
        
        # 使用前一周期数据计算当前周期的Pivot Points
        prev_high = high.shift(1)
        prev_low = low.shift(1)
        prev_close = close.shift(1)
        
        # Pivot Point
        df['pivot'] = (prev_high + prev_low + prev_close) / 3
        
        # 阻力位
        df['resistance_1'] = 2 * df['pivot'] - prev_low
        df['resistance_2'] = df['pivot'] + (prev_high - prev_low)
        df['resistance_3'] = prev_high + 2 * (df['pivot'] - prev_low)
        
        # 支撑位
        df['support_1'] = 2 * df['pivot'] - prev_high
        df['support_2'] = df['pivot'] - (prev_high - prev_low)
        df['support_3'] = prev_low - 2 * (prev_high - df['pivot'])
        
        return df
    
    def add_local_extrema(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        检测局部极值 (潜在支撑阻力位)
        """
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        
        # 检测局部最高点
        local_max_idx = argrelextrema(high, np.greater, order=self.window)[0]
        df['is_local_high'] = 0
        if len(local_max_idx) > 0:
            col_idx = df.columns.get_loc('is_local_high')
            for idx in local_max_idx:
                df.iloc[int(idx), int(col_idx)] = 1  # type: ignore
        
        # 检测局部最低点
        local_min_idx = argrelextrema(low, np.less, order=self.window)[0]
        df['is_local_low'] = 0
        if len(local_min_idx) > 0:
            col_idx = df.columns.get_loc('is_local_low')
            for idx in local_min_idx:
                df.iloc[int(idx), int(col_idx)] = 1  # type: ignore
        
        # 滚动窗口内的最近支撑阻力位
        window = self.window * 2
        
        # 最近N期内的最高价
        df['rolling_high'] = df['high'].rolling(window=window).max()
        # 最近N期内的最低价
        df['rolling_low'] = df['low'].rolling(window=window).min()
        
        return df
    
    def add_price_levels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        使用K-means聚类识别价格水平
        """
        if len(df) < self.n_clusters * 2:
            logger.warning("数据不足，跳过K-means聚类")
            for i in range(self.n_clusters):
                df[f'price_level_{i}'] = np.nan
            return df
        
        # 收集所有高低点
        high_values = np.asarray(df['high'].values, dtype=np.float64)
        low_values = np.asarray(df['low'].values, dtype=np.float64)
        prices = np.concatenate([high_values, low_values]).reshape(-1, 1)
        
        # 移除NaN
        prices = prices[~np.isnan(prices)]
        if len(prices) < self.n_clusters:
            for i in range(self.n_clusters):
                df[f'price_level_{i}'] = np.nan
            return df
        
        prices = prices.reshape(-1, 1)
        
        # K-means聚类
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        kmeans.fit(prices)
        
        # 获取聚类中心 (价格水平)
        levels = sorted(kmeans.cluster_centers_.flatten())
        
        for i, level in enumerate(levels):
            df[f'price_level_{i}'] = level
        
        return df
    
    def add_distance_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算价格到支撑阻力位的距离
        """
        close = df['close']
        
        # 到Pivot Points的距离 (百分比)
        if 'pivot' in df.columns:
            df['dist_to_pivot'] = (close - df['pivot']) / df['pivot']
            df['dist_to_r1'] = (close - df['resistance_1']) / df['resistance_1']
            df['dist_to_r2'] = (close - df['resistance_2']) / df['resistance_2']
            df['dist_to_s1'] = (close - df['support_1']) / df['support_1']
            df['dist_to_s2'] = (close - df['support_2']) / df['support_2']
        
        # 到滚动高低点的距离
        if 'rolling_high' in df.columns:
            df['dist_to_rolling_high'] = (close - df['rolling_high']) / df['rolling_high']
            df['dist_to_rolling_low'] = (close - df['rolling_low']) / df['rolling_low']
            
            # 在高低点范围内的相对位置 (0-1)
            df['price_position'] = (close - df['rolling_low']) / (
                df['rolling_high'] - df['rolling_low']
            ).replace(0, np.nan)
        
        # 到最近价格水平的距离
        price_level_cols = [c for c in df.columns if c.startswith('price_level_')]
        if price_level_cols:
            distances = []
            for col in price_level_cols:
                dist = abs(close - df[col]) / close
                distances.append(dist)
            
            df['dist_to_nearest_level'] = pd.concat(distances, axis=1).min(axis=1)
        
        return df
    
    def get_feature_names(self) -> list:
        """获取所有特征名称"""
        features = [
            # Pivot Points
            'pivot', 'resistance_1', 'resistance_2', 'resistance_3',
            'support_1', 'support_2', 'support_3',
            # 局部极值
            'is_local_high', 'is_local_low',
            'rolling_high', 'rolling_low',
            # 价格水平
            *[f'price_level_{i}' for i in range(self.n_clusters)],
            # 距离特征
            'dist_to_pivot', 'dist_to_r1', 'dist_to_r2',
            'dist_to_s1', 'dist_to_s2',
            'dist_to_rolling_high', 'dist_to_rolling_low',
            'price_position', 'dist_to_nearest_level'
        ]
        return features


# 全局实例
support_resistance = SupportResistance()
