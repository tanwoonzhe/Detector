"""
时序数据验证框架
================================
针对金融时序数据的特殊验证方法
- Purged K-Fold: 防止数据泄露
- Walk-Forward: 滚动前向验证
- Triple Barrier Labeling: 三重屏障标签

参考:
- López de Prado《Advances in Financial Machine Learning》
- CSDN/Kaggle上的时序验证最佳实践
"""

import numpy as np
import pandas as pd
from typing import Iterator, Tuple, List, Optional, Dict, Any, Union
from sklearn.model_selection import BaseCrossValidator
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import logging

logger = logging.getLogger(__name__)


class PurgedKFold(BaseCrossValidator):
    """
    Purged K-Fold交叉验证
    
    为金融时序数据设计，解决数据泄露问题:
    1. 时间顺序分割
    2. 训练集和验证集之间设置间隔(purge)
    3. 防止序列数据的信息泄露
    
    参考: Marcos López de Prado《Advances in Financial Machine Learning》Chapter 7
    """
    
    def __init__(
        self, 
        n_splits: int = 5, 
        purge_gap: int = 24,
        embargo_pct: float = 0.01
    ):
        """
        Args:
            n_splits: 折数
            purge_gap: 训练集和验证集之间的间隔(小时数)
            embargo_pct: 训练集末尾需要禁用的百分比
        """
        self.n_splits = n_splits
        self.purge_gap = purge_gap
        self.embargo_pct = embargo_pct
    
    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits
    
    def split(
        self, 
        X: Any, 
        y: Any = None, 
        groups: Any = None
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        生成训练/验证集索引
        
        Yields:
            (train_indices, val_indices)
        """
        n_samples = len(X)
        indices = np.arange(n_samples)
        
        # 每个验证集的大小
        fold_size = n_samples // self.n_splits
        
        for i in range(self.n_splits):
            # 验证集范围
            val_start = i * fold_size
            val_end = (i + 1) * fold_size if i < self.n_splits - 1 else n_samples
            
            # 应用purge间隔
            train_end = val_start - self.purge_gap
            
            # 训练集在验证集之前
            if train_end <= 0:
                continue
            
            # 应用embargo (禁用训练集末尾一小部分)
            embargo_size = int(train_end * self.embargo_pct)
            train_indices = indices[:train_end - embargo_size]
            val_indices = indices[val_start:val_end]
            
            if len(train_indices) > 0 and len(val_indices) > 0:
                yield train_indices, val_indices


class WalkForwardValidator:
    """
    Walk-Forward滚动验证
    
    模拟真实交易场景:
    1. 使用历史数据训练
    2. 在未来数据上测试
    3. 滚动窗口向前移动
    4. 支持扩张窗口或固定窗口
    """
    
    def __init__(
        self,
        train_size: int = 168 * 4,  # 4周训练数据 (小时)
        test_size: int = 168,       # 1周测试数据
        step_size: int = 24,        # 每次前进1天
        expanding: bool = True       # 是否使用扩张窗口
    ):
        """
        Args:
            train_size: 训练窗口大小
            test_size: 测试窗口大小
            step_size: 每次滚动的步长
            expanding: True=扩张窗口, False=固定窗口
        """
        self.train_size = train_size
        self.test_size = test_size
        self.step_size = step_size
        self.expanding = expanding
    
    def split(
        self, 
        X: np.ndarray
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        生成训练/测试集索引
        
        Yields:
            (train_indices, test_indices)
        """
        n_samples = len(X)
        indices = np.arange(n_samples)
        
        # 第一个测试窗口的起始位置
        test_start = self.train_size
        
        while test_start + self.test_size <= n_samples:
            # 测试集
            test_indices = indices[test_start:test_start + self.test_size]
            
            # 训练集
            if self.expanding:
                # 扩张窗口: 从头开始到测试集之前
                train_indices = indices[:test_start]
            else:
                # 固定窗口: 固定大小的训练窗口
                train_start = max(0, test_start - self.train_size)
                train_indices = indices[train_start:test_start]
            
            yield train_indices, test_indices
            
            # 滚动前进
            test_start += self.step_size
    
    def get_n_splits(self, X: np.ndarray) -> int:
        """计算分割数量"""
        n_samples = len(X)
        n_splits = 0
        test_start = self.train_size
        
        while test_start + self.test_size <= n_samples:
            n_splits += 1
            test_start += self.step_size
        
        return n_splits


class TripleBarrierLabeler:
    """
    三重屏障标签方法
    
    同时考虑:
    1. 止盈屏障 (Upper Barrier)
    2. 止损屏障 (Lower Barrier)
    3. 时间屏障 (Maximum Holding Period)
    
    参考: López de Prado《Advances in Financial Machine Learning》Chapter 3
    """
    
    def __init__(
        self,
        take_profit: float = 0.02,   # 止盈阈值 (2%)
        stop_loss: float = 0.02,     # 止损阈值 (2%)
        max_holding_period: int = 24, # 最大持有期 (小时)
        volatility_scaling: bool = True  # 是否根据波动率调整屏障
    ):
        """
        Args:
            take_profit: 止盈比例
            stop_loss: 止损比例
            max_holding_period: 最大持有期
            volatility_scaling: 是否使用波动率调整屏障高度
        """
        self.take_profit = take_profit
        self.stop_loss = stop_loss
        self.max_holding_period = max_holding_period
        self.volatility_scaling = volatility_scaling
    
    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        应用三重屏障标签
        
        Args:
            df: 包含'close'和可选'volatility'列的DataFrame
            
        Returns:
            添加了'barrier_label'和'barrier_time'列的DataFrame
        """
        df = df.copy()
        close = df['close'].values
        n = len(close)
        
        # 计算波动率用于动态调整屏障
        if self.volatility_scaling and 'volatility_24' not in df.columns:
            returns = pd.Series(close).pct_change()
            volatility = returns.rolling(24).std().values
        elif 'volatility_24' in df.columns:
            volatility = df['volatility_24'].values
        else:
            volatility = np.ones(n) * 0.01  # 默认1%波动率
        
        labels = np.zeros(n)
        touch_times = np.zeros(n)
        
        for i in range(n - 1):
            entry_price = close[i]
            
            # 动态屏障 (根据波动率调整)
            vol = volatility[i] if not np.isnan(volatility[i]) else 0.01
            if self.volatility_scaling:
                upper_barrier = entry_price * (1 + self.take_profit * vol * 10)
                lower_barrier = entry_price * (1 - self.stop_loss * vol * 10)
            else:
                upper_barrier = entry_price * (1 + self.take_profit)
                lower_barrier = entry_price * (1 - self.stop_loss)
            
            # 搜索触及屏障的时间
            label = 0
            touch_time = self.max_holding_period
            
            for j in range(1, min(self.max_holding_period + 1, n - i)):
                future_price = close[i + j]
                
                # 检查是否触及屏障
                if future_price >= upper_barrier:
                    label = 1  # 触及上轨 (看涨)
                    touch_time = j
                    break
                elif future_price <= lower_barrier:
                    label = -1  # 触及下轨 (看跌)
                    touch_time = j
                    break
            
            # 如果到达时间屏障还未触及价格屏障
            if label == 0 and i + self.max_holding_period < n:
                final_price = close[i + self.max_holding_period]
                if final_price > entry_price:
                    label = 1
                elif final_price < entry_price:
                    label = -1
            
            labels[i] = label
            touch_times[i] = touch_time
        
        df['barrier_label'] = labels
        df['barrier_time'] = touch_times
        
        return df


class TimeSeriesMetrics:
    """
    时序预测评估指标
    """
    
    @staticmethod
    def calculate_metrics(
        y_true: np.ndarray, 
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        计算分类指标
        
        Args:
            y_true: 真实标签
            y_pred: 预测标签
            y_proba: 预测概率 (可选)
            
        Returns:
            指标字典
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
            'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
        }
        
        # 每个类别的指标
        unique_classes = np.unique(y_true)
        for cls in unique_classes:
            cls_mask = y_true == cls
            if cls_mask.sum() > 0:
                metrics[f'precision_class_{cls}'] = precision_score(
                    y_true == cls, y_pred == cls, zero_division=0
                )
                metrics[f'recall_class_{cls}'] = recall_score(
                    y_true == cls, y_pred == cls, zero_division=0
                )
        
        return metrics
    
    @staticmethod
    def calculate_trading_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        returns: np.ndarray
    ) -> Dict[str, float]:
        """
        计算交易相关指标
        
        Args:
            y_true: 真实方向 (0=跌, 1=横盘, 2=涨)
            y_pred: 预测方向
            returns: 实际收益率
            
        Returns:
            交易指标
        """
        # 方向正确率 (忽略横盘预测)
        non_sideways_mask = y_pred != 1
        if non_sideways_mask.sum() > 0:
            direction_accuracy = accuracy_score(
                y_true[non_sideways_mask], 
                y_pred[non_sideways_mask]
            )
        else:
            direction_accuracy = 0
        
        # 模拟交易收益
        # 预测涨买入(+1), 预测跌卖出(-1), 横盘持有(0)
        positions = np.where(y_pred == 2, 1, np.where(y_pred == 0, -1, 0))
        strategy_returns = positions * returns
        
        # 累计收益
        cumulative_return = (1 + strategy_returns).prod() - 1
        
        # 夏普比率 (年化)
        if strategy_returns.std() > 0:
            sharpe = np.sqrt(24 * 365) * strategy_returns.mean() / strategy_returns.std()
        else:
            sharpe = 0
        
        # 最大回撤
        cumulative = (1 + strategy_returns).cumprod()
        rolling_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        return {
            'direction_accuracy': float(direction_accuracy),
            'cumulative_return': float(cumulative_return),
            'sharpe_ratio': float(sharpe),
            'max_drawdown': float(max_drawdown),
            'win_rate': float((strategy_returns > 0).mean())
        }
    
    @staticmethod
    def calculate_regression_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """
        计算回归指标
        """
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        return {
            'mse': float(mean_squared_error(y_true, y_pred)),
            'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
            'mae': float(mean_absolute_error(y_true, y_pred)),
            'r2': float(r2_score(y_true, y_pred)),
            'direction_accuracy': float(((y_true > 0) == (y_pred > 0)).mean())
        }
