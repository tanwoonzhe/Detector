"""
LightGBM 基准模型
================================
使用LightGBM作为快速基准模型

优势:
- 训练速度快
- 不需要GPU
- 可解释性好 (特征重要性)
- 适合表格数据

参考: Microsoft LightGBM文档
"""

import numpy as np
import lightgbm as lgb
from typing import Dict, Any, Tuple, List, Optional, cast
from pathlib import Path
import joblib
import logging

from .base import BasePredictor
from config import ModelConfig

logger = logging.getLogger(__name__)


class LightGBMPredictor(BasePredictor):
    """
    LightGBM分类器
    
    适用于:
    - 快速基准测试
    - 特征重要性分析
    - 集成学习的基模型
    """
    
    def __init__(
        self,
        n_estimators: int = 500,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        num_leaves: int = 31,
        min_child_samples: int = 20,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_alpha: float = 0.1,
        reg_lambda: float = 0.1,
        n_jobs: int = -1,
        **kwargs
    ):
        super().__init__(name="LightGBM", config=kwargs)
        
        self.params = {
            'objective': 'multiclass',
            'num_class': 3,
            'metric': 'multi_logloss',
            'boosting_type': 'gbdt',
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'num_leaves': num_leaves,
            'min_child_samples': min_child_samples,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'reg_alpha': reg_alpha,
            'reg_lambda': reg_lambda,
            'n_jobs': n_jobs,
            'verbose': -1,
            'random_state': 42
        }
        
        self.model: Optional[lgb.LGBMClassifier] = None
        self.feature_importance: Dict[str, float] = {}
    
    def build(self, input_shape: Optional[Tuple[int, ...]] = None, n_classes: int = 3) -> None:
        """构建模型"""
        self.params['num_class'] = n_classes
        self.model = lgb.LGBMClassifier(**self.params)
        logger.info(f"LightGBM模型构建完成")
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        训练模型
        
        注意: LightGBM使用扁平化的特征，不是序列
        对于序列数据，需要先展平或聚合
        """
        if self.model is None:
            self.build()
        
        assert self.model is not None, "模型未初始化"
        
        # 如果输入是3D (序列)，展平为2D
        if X_train.ndim == 3:
            X_train = self._flatten_sequences(X_train)
            if X_val is not None:
                X_val = self._flatten_sequences(X_val)
        
        # 训练参数
        callbacks = [
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=50)
        ]
        
        eval_set = [(X_val, y_val)] if X_val is not None else None
        
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            callbacks=callbacks
        )
        
        # 记录特征重要性
        if feature_names:
            importances = self.model.feature_importances_
            # 如果展平了序列，特征名需要相应扩展
            if len(importances) > len(feature_names):
                seq_len = len(importances) // len(feature_names)
                expanded_names = [
                    f"{name}_t{t}" 
                    for t in range(seq_len) 
                    for name in feature_names
                ]
                feature_names = expanded_names
            
            self.feature_importance = dict(zip(feature_names, importances))
        
        self._is_trained = True
        
        return {
            'best_iteration': self.model.best_iteration_,
            'best_score': self.model.best_score_
        }
    
    def _flatten_sequences(self, X: np.ndarray) -> np.ndarray:
        """
        展平序列数据
        
        Args:
            X: (samples, seq_len, features)
            
        Returns:
            X_flat: (samples, seq_len * features)
        """
        return X.reshape(X.shape[0], -1)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测类别"""
        if self.model is None:
            raise ValueError("模型未训练")
        if X.ndim == 3:
            X = self._flatten_sequences(X)
        return cast(np.ndarray, self.model.predict(X))
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """预测概率"""
        if self.model is None:
            raise ValueError("模型未训练")
        if X.ndim == 3:
            X = self._flatten_sequences(X)
        return cast(np.ndarray, self.model.predict_proba(X))
    
    def get_feature_importance(self, top_n: int = 20) -> Dict[str, float]:
        """获取Top N重要特征"""
        sorted_importance = sorted(
            self.feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return dict(sorted_importance[:top_n])
    
    def save(self, path: Path) -> None:
        """保存模型"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        joblib.dump({
            'model': self.model,
            'params': self.params,
            'feature_importance': self.feature_importance
        }, path)
        logger.info(f"LightGBM模型已保存: {path}")
    
    def load(self, path: Path) -> None:
        """加载模型"""
        path = Path(path)
        data = joblib.load(path)
        
        self.model = data['model']
        self.params = data['params']
        self.feature_importance = data.get('feature_importance', {})
        self._is_trained = True
        
        logger.info(f"LightGBM模型已加载: {path}")


class SequenceLightGBM(LightGBMPredictor):
    """
    针对序列数据优化的LightGBM
    
    使用聚合特征而非简单展平:
    - 时序特征的统计量 (均值、标准差、最大最小等)
    - 最近N个时刻的原始值
    """
    
    def __init__(self, recent_steps: int = 24, **kwargs):
        """
        Args:
            recent_steps: 保留最近N个时刻的原始值
        """
        super().__init__(**kwargs)
        self.recent_steps = recent_steps
        self.name = "SequenceLightGBM"
    
    def _extract_sequence_features(self, X: np.ndarray) -> np.ndarray:
        """
        从序列中提取聚合特征
        
        Args:
            X: (samples, seq_len, features)
            
        Returns:
            X_agg: (samples, aggregated_features)
        """
        samples, seq_len, n_features = X.shape
        
        features = []
        
        # 统计特征
        features.append(np.mean(X, axis=1))  # 均值
        features.append(np.std(X, axis=1))   # 标准差
        features.append(np.max(X, axis=1))   # 最大值
        features.append(np.min(X, axis=1))   # 最小值
        
        # 最近N步的原始值
        recent = min(self.recent_steps, seq_len)
        features.append(X[:, -recent:, :].reshape(samples, -1))
        
        # 变化率特征
        diff = np.diff(X, axis=1)
        features.append(np.mean(diff, axis=1))  # 平均变化
        features.append(np.std(diff, axis=1))   # 变化波动
        
        # 趋势特征 (简单线性回归斜率)
        t = np.arange(seq_len)
        slopes = np.zeros((samples, n_features))
        for i in range(samples):
            for j in range(n_features):
                slopes[i, j] = np.polyfit(t, X[i, :, j], 1)[0]
        features.append(slopes)
        
        return np.hstack(features)
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """训练"""
        if X_train.ndim == 3:
            X_train = self._extract_sequence_features(X_train)
            if X_val is not None:
                X_val = self._extract_sequence_features(X_val)
        
        return super().train(X_train, y_train, X_val, y_val, feature_names)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 3:
            X = self._extract_sequence_features(X)
        return super().predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 3:
            X = self._extract_sequence_features(X)
        return super().predict_proba(X)
