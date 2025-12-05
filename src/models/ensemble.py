"""
模型集成框架
================================
整合多个模型的预测结果

集成策略:
1. 简单投票 (Hard Voting)
2. 加权平均 (Soft Voting)
3. 堆叠 (Stacking)
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import logging

from .base import BasePredictor

logger = logging.getLogger(__name__)


class ModelEnsemble:
    """
    模型集成类
    
    支持多种集成策略
    """
    
    def __init__(
        self,
        models: Optional[List[BasePredictor]] = None,
        weights: Optional[List[float]] = None,
        strategy: str = 'soft_voting'
    ):
        """
        Args:
            models: 基模型列表
            weights: 各模型权重 (None=等权)
            strategy: 集成策略 ('hard_voting', 'soft_voting', 'best_model')
        """
        self.models = models or []
        self.weights = weights
        self.strategy = strategy
        self._is_trained = False
    
    def add_model(self, model: BasePredictor, weight: float = 1.0) -> None:
        """添加模型"""
        self.models.append(model)
        if self.weights is None:
            self.weights = [1.0] * len(self.models)
        else:
            self.weights.append(weight)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        集成预测
        
        Args:
            X: 输入特征
            
        Returns:
            predictions: 预测类别
        """
        if self.strategy == 'hard_voting':
            return self._hard_voting(X)
        elif self.strategy == 'soft_voting':
            return self._soft_voting(X)
        elif self.strategy == 'best_model':
            return self._best_model(X)
        else:
            raise ValueError(f"未知策略: {self.strategy}")
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        集成概率预测
        
        Returns:
            probabilities: (samples, n_classes)
        """
        # 收集所有模型的概率预测
        all_proba = []
        for model in self.models:
            if model.is_trained:
                proba = model.predict_proba(X)
                all_proba.append(proba)
        
        if not all_proba:
            raise ValueError("没有已训练的模型")
        
        # 加权平均
        weights = self._normalize_weights()
        weights = weights[:len(all_proba)]
        weights = weights / weights.sum()
        
        weighted_proba = np.zeros_like(all_proba[0])
        for proba, w in zip(all_proba, weights):
            weighted_proba += proba * w
        
        return weighted_proba
    
    def _hard_voting(self, X: np.ndarray) -> np.ndarray:
        """硬投票"""
        all_preds = []
        for model in self.models:
            if model.is_trained:
                preds = model.predict(X)
                all_preds.append(preds)
        
        if not all_preds:
            raise ValueError("没有已训练的模型")
        
        # 多数投票
        all_preds = np.array(all_preds)  # (n_models, n_samples)
        
        # 对每个样本，统计各类别的票数
        n_samples = all_preds.shape[1]
        final_preds = np.zeros(n_samples, dtype=int)
        
        for i in range(n_samples):
            votes = np.bincount(all_preds[:, i].astype(int), minlength=3)
            final_preds[i] = np.argmax(votes)
        
        return final_preds
    
    def _soft_voting(self, X: np.ndarray) -> np.ndarray:
        """软投票 (概率加权平均)"""
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)
    
    def _best_model(self, X: np.ndarray) -> np.ndarray:
        """使用最佳模型 (权重最高的模型)"""
        weights = self._normalize_weights()
        best_idx = np.argmax(weights)
        return self.models[best_idx].predict(X)
    
    def _normalize_weights(self) -> np.ndarray:
        """归一化权重"""
        if self.weights is None:
            return np.ones(len(self.models)) / len(self.models)
        weights = np.array(self.weights[:len(self.models)])
        return weights / weights.sum()
    
    def get_model_contributions(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        获取各模型的贡献
        
        Returns:
            {model_name: predictions}
        """
        contributions = {}
        for model in self.models:
            if model.is_trained:
                contributions[model.name] = model.predict(X)
        return contributions
    
    def update_weights(self, val_scores: Dict[str, float]) -> None:
        """
        根据验证集性能更新权重
        
        Args:
            val_scores: {model_name: accuracy}
        """
        new_weights = []
        for model in self.models:
            score = val_scores.get(model.name, 0.5)
            # 使用准确率作为权重
            new_weights.append(score)
        
        self.weights = new_weights
        logger.info(f"更新集成权重: {dict(zip([m.name for m in self.models], new_weights))}")


class StackingEnsemble:
    """
    堆叠集成
    
    使用元学习器整合基模型的预测
    """
    
    def __init__(
        self,
        base_models: List[BasePredictor],
        meta_model: Optional[BasePredictor] = None
    ):
        """
        Args:
            base_models: 基模型列表
            meta_model: 元学习器 (默认使用LightGBM)
        """
        self.base_models = base_models
        
        if meta_model is None:
            from .lightgbm_model import LightGBMPredictor
            self.meta_model = LightGBMPredictor(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1
            )
        else:
            self.meta_model = meta_model
    
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        训练堆叠模型
        
        1. 训练基模型
        2. 生成基模型预测作为元特征
        3. 训练元模型
        """
        logger.info("训练堆叠集成...")
        
        # 分割数据用于生成元特征 (避免信息泄露)
        split_idx = int(len(X_train) * 0.7)
        X_base = X_train[:split_idx]
        y_base = y_train[:split_idx]
        X_meta = X_train[split_idx:]
        y_meta = y_train[split_idx:]
        
        # 训练基模型
        for model in self.base_models:
            logger.info(f"  训练基模型: {model.name}")
            if not model._is_trained:
                if hasattr(model, 'build') and model.model is None:
                    input_shape = (X_base.shape[1], X_base.shape[2]) if X_base.ndim == 3 else X_base.shape[1:]
                    model.build(input_shape)
                model.train(X_base, y_base, X_val, y_val)
        
        # 生成元特征
        meta_features = self._generate_meta_features(X_meta)
        
        # 训练元模型
        logger.info(f"  训练元模型: {self.meta_model.name}")
        self.meta_model.build(meta_features.shape[1:])
        self.meta_model.train(meta_features, y_meta, None, None)
        
        return {'status': 'success'}
    
    def _generate_meta_features(self, X: np.ndarray) -> np.ndarray:
        """生成元特征"""
        meta_features = []
        
        for model in self.base_models:
            if model.is_trained:
                proba = model.predict_proba(X)
                meta_features.append(proba)
        
        # 拼接所有基模型的概率预测
        return np.hstack(meta_features)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测"""
        meta_features = self._generate_meta_features(X)
        return self.meta_model.predict(meta_features)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """概率预测"""
        meta_features = self._generate_meta_features(X)
        return self.meta_model.predict_proba(meta_features)


class MultiWindowEnsemble:
    """
    多窗口预测集成
    
    整合多个预测窗口的结果
    """
    
    def __init__(self, prediction_windows: List[float] = [0.5, 1, 2, 4]):
        self.prediction_windows = prediction_windows
        self.predictions: Dict[float, np.ndarray] = {}
        self.probabilities: Dict[float, np.ndarray] = {}
    
    def set_predictions(
        self, 
        window: float, 
        predictions: np.ndarray,
        probabilities: Optional[np.ndarray] = None
    ) -> None:
        """设置某窗口的预测"""
        self.predictions[window] = predictions
        if probabilities is not None:
            self.probabilities[window] = probabilities
    
    def get_consensus(self) -> np.ndarray:
        """
        获取各窗口的共识预测
        
        如果多数窗口预测相同方向，则返回该方向
        否则返回横盘(1)
        """
        if not self.predictions:
            raise ValueError("没有预测数据")
        
        n_samples = len(list(self.predictions.values())[0])
        consensus = np.ones(n_samples, dtype=int)  # 默认横盘
        
        for i in range(n_samples):
            votes = [self.predictions[w][i] for w in self.predictions]
            vote_counts = np.bincount(votes, minlength=3)
            
            # 需要超过半数一致才采用
            if vote_counts.max() > len(votes) / 2:
                consensus[i] = vote_counts.argmax()
        
        return consensus
    
    def get_weighted_consensus(
        self, 
        window_weights: Optional[Dict[float, float]] = None
    ) -> np.ndarray:
        """
        加权共识
        
        Args:
            window_weights: 各窗口权重，默认短期权重高
        """
        if window_weights is None:
            # 默认: 短期窗口权重更高
            window_weights = {0.5: 0.35, 1: 0.30, 2: 0.20, 4: 0.15}
        
        if not self.probabilities:
            return self.get_consensus()
        
        n_samples = len(list(self.probabilities.values())[0])
        weighted_proba = np.zeros((n_samples, 3))
        
        for window, proba in self.probabilities.items():
            weight = window_weights.get(window, 1.0 / len(self.probabilities))
            weighted_proba += proba * weight
        
        return np.argmax(weighted_proba, axis=1)
    
    def get_confidence(self) -> np.ndarray:
        """
        获取预测置信度
        
        基于窗口间的一致性
        """
        if not self.predictions:
            raise ValueError("没有预测数据")
        
        n_samples = len(list(self.predictions.values())[0])
        confidence = np.zeros(n_samples)
        
        for i in range(n_samples):
            votes = [self.predictions[w][i] for w in self.predictions]
            vote_counts = np.bincount(votes, minlength=3)
            # 一致性比例作为置信度
            confidence[i] = vote_counts.max() / len(votes)
        
        return confidence
