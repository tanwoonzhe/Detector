"""
蜡烛图形态检测模块
================================
检测常见的日本蜡烛图形态
参考: Steve Nison《Japanese Candlestick Charting Techniques》
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class CandlestickPatterns:
    """
    蜡烛图形态检测类
    
    检测形态:
    - 单蜡烛: 十字星, 锤子线, 上吊线, 倒锤子, 射击之星
    - 双蜡烛: 吞没形态, 乌云盖顶, 刺透形态
    - 三蜡烛: 晨星, 暮星, 三白兵, 三黑鸦
    """
    
    def __init__(self, body_threshold: float = 0.1, shadow_threshold: float = 2.0):
        """
        Args:
            body_threshold: 小实体阈值 (相对于日内波幅)
            shadow_threshold: 长影线阈值 (影线长度/实体长度)
        """
        self.body_threshold = body_threshold
        self.shadow_threshold = shadow_threshold
    
    def add_all_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        添加所有蜡烛图形态特征
        
        Args:
            df: 必须包含 open, high, low, close 列
            
        Returns:
            添加形态后的DataFrame
        """
        df = df.copy()
        
        # 计算基础蜡烛属性
        df = self._calculate_candle_properties(df)
        
        # 单蜡烛形态
        df = self._detect_single_patterns(df)
        
        # 双蜡烛形态
        df = self._detect_double_patterns(df)
        
        # 三蜡烛形态
        df = self._detect_triple_patterns(df)
        
        # 综合信号
        df = self._calculate_pattern_signals(df)
        
        logger.info("蜡烛图形态检测完成")
        return df
    
    def _calculate_candle_properties(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算蜡烛属性"""
        open_price = df['open']
        high = df['high']
        low = df['low']
        close = df['close']
        
        # 实体
        df['body'] = close - open_price
        df['body_abs'] = abs(df['body'])
        
        # 日内波幅
        df['range'] = high - low
        
        # 上下影线
        df['upper_shadow'] = high - np.maximum(open_price, close)
        df['lower_shadow'] = np.minimum(open_price, close) - low
        
        # 实体中心
        df['body_center'] = (open_price + close) / 2
        
        # 蜡烛颜色 (1=阳线, -1=阴线, 0=十字)
        df['candle_color'] = np.where(
            close > open_price, 1,
            np.where(close < open_price, -1, 0)
        )
        
        # 相对实体大小 (实体/波幅)
        df['body_ratio'] = df['body_abs'] / df['range'].replace(0, np.nan)
        
        return df
    
    def _detect_single_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """检测单蜡烛形态"""
        
        # 十字星 (Doji): 实体很小
        df['doji'] = (df['body_ratio'] < self.body_threshold).astype(int)
        
        # 锤子线 (Hammer): 下影线长, 实体在顶部, 阳线
        hammer_cond = (
            (df['lower_shadow'] > df['body_abs'] * self.shadow_threshold) &
            (df['upper_shadow'] < df['body_abs'] * 0.5) &
            (df['candle_color'] >= 0)
        )
        df['hammer'] = hammer_cond.astype(int)
        
        # 上吊线 (Hanging Man): 与锤子线形状相同，但出现在上涨后
        # 需要结合趋势判断，这里先标记形状
        df['hanging_man'] = hammer_cond.astype(int)
        
        # 倒锤子 (Inverted Hammer): 上影线长, 实体在底部
        inverted_hammer_cond = (
            (df['upper_shadow'] > df['body_abs'] * self.shadow_threshold) &
            (df['lower_shadow'] < df['body_abs'] * 0.5) &
            (df['candle_color'] >= 0)
        )
        df['inverted_hammer'] = inverted_hammer_cond.astype(int)
        
        # 射击之星 (Shooting Star): 与倒锤子形状相同
        df['shooting_star'] = inverted_hammer_cond.astype(int)
        
        # 大阳线 (Marubozu Bullish): 几乎没有影线
        df['marubozu_bullish'] = (
            (df['candle_color'] == 1) &
            (df['body_ratio'] > 0.9)
        ).astype(int)
        
        # 大阴线 (Marubozu Bearish)
        df['marubozu_bearish'] = (
            (df['candle_color'] == -1) &
            (df['body_ratio'] > 0.9)
        ).astype(int)
        
        return df
    
    def _detect_double_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """检测双蜡烛形态"""
        
        # 前一根蜡烛数据
        prev_open = df['open'].shift(1)
        prev_close = df['close'].shift(1)
        prev_color = df['candle_color'].shift(1)
        prev_body = df['body_abs'].shift(1)
        
        curr_open = df['open']
        curr_close = df['close']
        curr_color = df['candle_color']
        curr_body = df['body_abs']
        
        # 看涨吞没 (Bullish Engulfing): 阴线后阳线, 阳线实体完全包住阴线
        bullish_engulfing = (
            (prev_color == -1) &
            (curr_color == 1) &
            (curr_open <= prev_close) &
            (curr_close >= prev_open)
        )
        df['bullish_engulfing'] = bullish_engulfing.astype(int)
        
        # 看跌吞没 (Bearish Engulfing)
        bearish_engulfing = (
            (prev_color == 1) &
            (curr_color == -1) &
            (curr_open >= prev_close) &
            (curr_close <= prev_open)
        )
        df['bearish_engulfing'] = bearish_engulfing.astype(int)
        
        # 乌云盖顶 (Dark Cloud Cover): 阳线后阴线, 阴线开盘高于阳线最高, 收盘在阳线中部以下
        prev_high = df['high'].shift(1)
        prev_mid = (prev_open + prev_close) / 2
        
        dark_cloud = (
            (prev_color == 1) &
            (curr_color == -1) &
            (curr_open > prev_high) &
            (curr_close < prev_mid) &
            (curr_close > prev_open)
        )
        df['dark_cloud_cover'] = dark_cloud.astype(int)
        
        # 刺透形态 (Piercing Pattern): 乌云盖顶的反转
        prev_low = df['low'].shift(1)
        
        piercing = (
            (prev_color == -1) &
            (curr_color == 1) &
            (curr_open < prev_low) &
            (curr_close > prev_mid) &
            (curr_close < prev_open)
        )
        df['piercing_pattern'] = piercing.astype(int)
        
        return df
    
    def _detect_triple_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """检测三蜡烛形态"""
        
        # 前两根蜡烛数据
        color_2 = df['candle_color'].shift(2)
        color_1 = df['candle_color'].shift(1)
        color_0 = df['candle_color']
        
        close_2 = df['close'].shift(2)
        close_1 = df['close'].shift(1)
        close_0 = df['close']
        
        open_2 = df['open'].shift(2)
        open_1 = df['open'].shift(1)
        open_0 = df['open']
        
        body_2 = df['body_abs'].shift(2)
        body_1 = df['body_abs'].shift(1)
        body_0 = df['body_abs']
        
        doji_1 = df['doji'].shift(1)
        
        # 晨星 (Morning Star): 阴线 + 小实体/十字星 + 阳线
        morning_star = (
            (color_2 == -1) &
            (body_1 < body_2 * 0.3) &  # 中间小实体
            (color_0 == 1) &
            (close_0 > (open_2 + close_2) / 2)  # 阳线收盘高于第一根中点
        )
        df['morning_star'] = morning_star.astype(int)
        
        # 暮星 (Evening Star): 阳线 + 小实体/十字星 + 阴线
        evening_star = (
            (color_2 == 1) &
            (body_1 < body_2 * 0.3) &
            (color_0 == -1) &
            (close_0 < (open_2 + close_2) / 2)
        )
        df['evening_star'] = evening_star.astype(int)
        
        # 三白兵 (Three White Soldiers): 连续三根阳线, 每根收盘更高
        three_white_soldiers = (
            (color_2 == 1) &
            (color_1 == 1) &
            (color_0 == 1) &
            (close_1 > close_2) &
            (close_0 > close_1)
        )
        df['three_white_soldiers'] = three_white_soldiers.astype(int)
        
        # 三黑鸦 (Three Black Crows): 连续三根阴线, 每根收盘更低
        three_black_crows = (
            (color_2 == -1) &
            (color_1 == -1) &
            (color_0 == -1) &
            (close_1 < close_2) &
            (close_0 < close_1)
        )
        df['three_black_crows'] = three_black_crows.astype(int)
        
        return df
    
    def _calculate_pattern_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算综合形态信号"""
        
        # 看涨形态信号
        bullish_patterns = [
            'hammer', 'inverted_hammer', 'marubozu_bullish',
            'bullish_engulfing', 'piercing_pattern',
            'morning_star', 'three_white_soldiers'
        ]
        
        df['bullish_pattern_count'] = sum(
            df[p] for p in bullish_patterns if p in df.columns
        )
        
        # 看跌形态信号
        bearish_patterns = [
            'hanging_man', 'shooting_star', 'marubozu_bearish',
            'bearish_engulfing', 'dark_cloud_cover',
            'evening_star', 'three_black_crows'
        ]
        
        df['bearish_pattern_count'] = sum(
            df[p] for p in bearish_patterns if p in df.columns
        )
        
        # 综合信号 (-N 到 +N)
        df['pattern_signal'] = df['bullish_pattern_count'] - df['bearish_pattern_count']
        
        return df
    
    def get_pattern_names(self) -> list:
        """获取所有形态名称"""
        return [
            # 基础属性
            'body', 'body_abs', 'range', 'upper_shadow', 'lower_shadow',
            'body_center', 'candle_color', 'body_ratio',
            # 单蜡烛
            'doji', 'hammer', 'hanging_man', 'inverted_hammer',
            'shooting_star', 'marubozu_bullish', 'marubozu_bearish',
            # 双蜡烛
            'bullish_engulfing', 'bearish_engulfing',
            'dark_cloud_cover', 'piercing_pattern',
            # 三蜡烛
            'morning_star', 'evening_star',
            'three_white_soldiers', 'three_black_crows',
            # 综合
            'bullish_pattern_count', 'bearish_pattern_count', 'pattern_signal'
        ]


# 全局实例
candlestick_patterns = CandlestickPatterns()
