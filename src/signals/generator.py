"""
交易信号生成器
================================
整合模型预测，生成买卖信号

信号类型:
- 0: 卖出信号 (预测下跌)
- 1: 观望 (横盘/不确定)
- 2: 买入信号 (预测上涨)
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import IntEnum
from datetime import datetime
import logging

from config import SignalConfig, TradingConfig

logger = logging.getLogger(__name__)


class SignalType(IntEnum):
    """信号类型"""
    SELL = 0      # 卖出
    HOLD = 1      # 观望
    BUY = 2       # 买入


@dataclass
class TradingSignal:
    """交易信号数据类"""
    timestamp: datetime
    signal: SignalType
    confidence: float           # 置信度 0-1
    direction: str              # "bullish", "bearish", "neutral"
    magnitude: float            # 预测幅度
    predictions: Dict[float, int]  # 各窗口预测 {window: prediction}
    window_confidences: Dict[float, float]  # 各窗口置信度
    sentiment_score: float      # 情感得分
    technical_score: float      # 技术面得分
    summary: str                # 信号摘要


class SignalGenerator:
    """
    信号生成器
    
    整合:
    - 模型预测
    - 多窗口一致性
    - 情感分析
    - 技术指标
    """
    
    def __init__(
        self,
        min_confidence: Optional[float] = None,
        prediction_windows: Optional[List[float]] = None,
        window_weights: Optional[Dict[float, float]] = None
    ):
        self.min_confidence = min_confidence if min_confidence is not None else getattr(SignalConfig, 'MIN_CONFIDENCE', 0.5)
        self.prediction_windows = prediction_windows or TradingConfig.PREDICTION_WINDOWS
        
        # 窗口权重 (短期权重更高)
        self.window_weights = window_weights or {
            0.5: 0.35,
            1.0: 0.30,
            2.0: 0.20,
            4.0: 0.15
        }
    
    def generate_signal(
        self,
        predictions: Dict[float, np.ndarray],
        probabilities: Dict[float, np.ndarray],
        sentiment_score: float = 0.0,
        technical_indicators: Optional[Dict[str, float]] = None,
        timestamp: Optional[datetime] = None
    ) -> TradingSignal:
        """
        生成单个时间点的交易信号
        
        Args:
            predictions: {window: prediction_array} - 各窗口预测
            probabilities: {window: probability_array} - 各窗口概率
            sentiment_score: 情感得分 (-1 到 1)
            technical_indicators: 技术指标字典
            timestamp: 时间戳
            
        Returns:
            TradingSignal
        """
        timestamp = timestamp or datetime.now()
        technical_indicators = technical_indicators or {}
        
        # 提取最新预测 (取最后一个)
        latest_preds = {w: int(p[-1]) if len(p) > 0 else 1 for w, p in predictions.items()}
        latest_probs = {w: p[-1] if len(p) > 0 else np.array([0.33, 0.34, 0.33]) 
                       for w, p in probabilities.items()}
        
        # 计算加权共识
        weighted_proba = np.zeros(3)
        for window in self.prediction_windows:
            if window in latest_probs:
                weight = self.window_weights.get(window, 0.25)
                weighted_proba += latest_probs[window] * weight
        
        # 主信号
        main_signal = SignalType(np.argmax(weighted_proba))
        
        # 置信度 (基于概率和窗口一致性)
        prob_confidence = weighted_proba.max()
        
        # 窗口一致性
        pred_values = list(latest_preds.values())
        consistency = max(pred_values.count(0), pred_values.count(1), pred_values.count(2)) / len(pred_values)
        
        # 综合置信度
        confidence = 0.6 * prob_confidence + 0.4 * consistency
        
        # 各窗口置信度
        window_confidences = {w: float(p.max()) for w, p in latest_probs.items()}
        
        # 计算预测幅度 (基于概率差)
        if main_signal == SignalType.BUY:
            magnitude = weighted_proba[2] - weighted_proba[0]
        elif main_signal == SignalType.SELL:
            magnitude = weighted_proba[0] - weighted_proba[2]
        else:
            magnitude = 0.0
        
        # 计算技术面得分
        technical_score = self._calculate_technical_score(technical_indicators)
        
        # 综合调整
        signal, confidence = self._adjust_signal(
            main_signal, confidence, sentiment_score, technical_score
        )
        
        # 确定方向描述
        if signal == SignalType.BUY:
            direction = "bullish"
        elif signal == SignalType.SELL:
            direction = "bearish"
        else:
            direction = "neutral"
        
        # 生成摘要
        summary = self._generate_summary(
            signal, confidence, latest_preds, sentiment_score, technical_score
        )
        
        return TradingSignal(
            timestamp=timestamp,
            signal=signal,
            confidence=confidence,
            direction=direction,
            magnitude=magnitude,
            predictions=latest_preds,
            window_confidences=window_confidences,
            sentiment_score=sentiment_score,
            technical_score=technical_score,
            summary=summary
        )
    
    def _calculate_technical_score(self, indicators: Dict[str, float]) -> float:
        """
        计算技术面得分 (-1 到 1)
        
        综合多个技术指标
        """
        if not indicators:
            return 0.0
        
        scores = []
        
        # RSI得分
        if 'rsi' in indicators:
            rsi = indicators['rsi']
            if rsi > 70:
                scores.append(-1.0)  # 超买
            elif rsi < 30:
                scores.append(1.0)   # 超卖
            else:
                scores.append((50 - rsi) / 50 * 0.5)
        
        # MACD得分
        if 'macd_histogram' in indicators:
            macd_hist = indicators['macd_histogram']
            if macd_hist > 0:
                scores.append(0.5)
            else:
                scores.append(-0.5)
        
        # 布林带位置
        if 'bb_position' in indicators:
            bb_pos = indicators['bb_position']
            if bb_pos > 0.8:
                scores.append(-0.5)  # 接近上轨
            elif bb_pos < 0.2:
                scores.append(0.5)   # 接近下轨
            else:
                scores.append(0)
        
        # 趋势方向
        if 'trend_direction' in indicators:
            scores.append(indicators['trend_direction'] * 0.5)
        
        return float(np.mean(scores)) if scores else 0.0
    
    def _adjust_signal(
        self,
        signal: SignalType,
        confidence: float,
        sentiment: float,
        technical: float
    ) -> Tuple[SignalType, float]:
        """
        根据情感和技术指标调整信号
        """
        # 情感和技术的加权
        adjustment = 0.3 * sentiment + 0.2 * technical
        
        # 如果信号与情感/技术面一致，增加置信度
        if signal == SignalType.BUY and adjustment > 0:
            confidence = min(1.0, confidence + abs(adjustment) * 0.1)
        elif signal == SignalType.SELL and adjustment < 0:
            confidence = min(1.0, confidence + abs(adjustment) * 0.1)
        # 如果不一致，降低置信度
        elif signal == SignalType.BUY and adjustment < -0.3:
            confidence *= 0.8
        elif signal == SignalType.SELL and adjustment > 0.3:
            confidence *= 0.8
        
        # 置信度太低则改为观望
        if confidence < self.min_confidence:
            signal = SignalType.HOLD
        
        return signal, confidence
    
    def _generate_summary(
        self,
        signal: SignalType,
        confidence: float,
        predictions: Dict[float, int],
        sentiment: float,
        technical: float
    ) -> str:
        """生成信号摘要"""
        # 信号描述
        if signal == SignalType.BUY:
            signal_desc = "📈 买入信号"
        elif signal == SignalType.SELL:
            signal_desc = "📉 卖出信号"
        else:
            signal_desc = "⏸️ 观望"
        
        # 置信度级别
        if confidence >= 0.8:
            conf_desc = "强"
        elif confidence >= 0.6:
            conf_desc = "中"
        else:
            conf_desc = "弱"
        
        # 窗口一致性
        window_desc = []
        for w, p in predictions.items():
            direction = "↑" if p == 2 else ("↓" if p == 0 else "→")
            window_desc.append(f"{w}h:{direction}")
        
        # 情感描述
        if sentiment > 0.3:
            sent_desc = "情感偏多"
        elif sentiment < -0.3:
            sent_desc = "情感偏空"
        else:
            sent_desc = "情感中性"
        
        # 技术面描述
        if technical > 0.3:
            tech_desc = "技术面看涨"
        elif technical < -0.3:
            tech_desc = "技术面看跌"
        else:
            tech_desc = "技术面中性"
        
        summary = (
            f"{signal_desc} ({conf_desc}信号, 置信度{confidence:.1%})\n"
            f"预测: {', '.join(window_desc)}\n"
            f"{sent_desc} | {tech_desc}"
        )
        
        return summary
    
    def generate_batch_signals(
        self,
        predictions: Dict[float, np.ndarray],
        probabilities: Dict[float, np.ndarray],
        timestamps: Optional[List[datetime]] = None,
        sentiment_scores: Optional[np.ndarray] = None,
        technical_df: Optional[pd.DataFrame] = None
    ) -> List[TradingSignal]:
        """
        批量生成信号
        
        Args:
            predictions: {window: predictions_array}
            probabilities: {window: probabilities_array}
            timestamps: 时间戳列表
            sentiment_scores: 情感得分数组
            technical_df: 技术指标DataFrame
            
        Returns:
            信号列表
        """
        # 确定样本数
        n_samples = len(list(predictions.values())[0])
        
        if timestamps is None:
            timestamps = [datetime.now()] * n_samples
        
        if sentiment_scores is None:
            sentiment_scores = np.zeros(n_samples)
        
        signals = []
        for i in range(n_samples):
            # 提取当前时间点的数据
            curr_preds = {w: p[i:i+1] for w, p in predictions.items()}
            curr_probs = {w: p[i:i+1] for w, p in probabilities.items()}
            
            sentiment = sentiment_scores[i] if sentiment_scores is not None and i < len(sentiment_scores) else 0.0
            
            tech_indicators: Dict[str, float] = {}
            if technical_df is not None and i < len(technical_df):
                row = technical_df.iloc[i]
                for col in ['rsi', 'macd_histogram', 'bb_position', 'trend_direction']:
                    if col in row:
                        tech_indicators[col] = row[col]
            
            ts = timestamps[i] if timestamps is not None and i < len(timestamps) else datetime.now()
            signal = self.generate_signal(
                curr_preds, curr_probs,
                sentiment, tech_indicators,
                ts
            )
            signals.append(signal)
        
        return signals


class SignalFormatter:
    """信号格式化输出"""
    
    @staticmethod
    def to_dict(signal: TradingSignal) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'timestamp': signal.timestamp.isoformat(),
            'signal': signal.signal.name,
            'signal_value': int(signal.signal),
            'confidence': signal.confidence,
            'direction': signal.direction,
            'magnitude': signal.magnitude,
            'predictions': signal.predictions,
            'window_confidences': signal.window_confidences,
            'sentiment_score': signal.sentiment_score,
            'technical_score': signal.technical_score,
            'summary': signal.summary
        }
    
    @staticmethod
    def to_dataframe(signals: List[TradingSignal]) -> pd.DataFrame:
        """转换为DataFrame"""
        records = [SignalFormatter.to_dict(s) for s in signals]
        return pd.DataFrame(records)
    
    @staticmethod
    def format_display(signal: TradingSignal) -> str:
        """格式化显示"""
        emoji = "📈" if signal.signal == SignalType.BUY else (
            "📉" if signal.signal == SignalType.SELL else "⏸️"
        )
        
        display = f"""
╔══════════════════════════════════════════╗
║  {emoji} {signal.signal.name} - 置信度: {signal.confidence:.1%}
╠══════════════════════════════════════════╣
║  时间: {signal.timestamp.strftime('%Y-%m-%d %H:%M')}
║  方向: {signal.direction.upper()}
║  预测幅度: {signal.magnitude:.2%}
╠══════════════════════════════════════════╣
║  各窗口预测:
"""
        for w, p in signal.predictions.items():
            direction = "上涨" if p == 2 else ("下跌" if p == 0 else "横盘")
            conf = signal.window_confidences.get(w, 0)
            display += f"║    {w}小时: {direction} ({conf:.1%})\n"
        
        display += f"""╠══════════════════════════════════════════╣
║  情感得分: {signal.sentiment_score:+.2f}
║  技术得分: {signal.technical_score:+.2f}
╚══════════════════════════════════════════╝
"""
        return display
