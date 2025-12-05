"""
模型基类定义
================================
所有预测模型的抽象基类
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional, Sized, cast
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class BasePredictor(ABC):
    """
    预测模型基类
    
    所有模型必须实现:
    - build(): 构建模型
    - train(): 训练模型  
    - predict(): 预测
    - save()/load(): 保存/加载模型
    """
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        self.name = name
        self.config = config or {}
        self.model = None
        self._is_trained = False
    
    @abstractmethod
    def build(self, input_shape: Tuple[int, ...], n_classes: int = 3) -> None:
        """构建模型"""
        pass
    
    @abstractmethod
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """训练模型"""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测类别"""
        pass
    
    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """预测概率"""
        pass
    
    @abstractmethod
    def save(self, path: Path) -> None:
        """保存模型"""
        pass
    
    @abstractmethod
    def load(self, path: Path) -> None:
        """加载模型"""
        pass
    
    @property
    def is_trained(self) -> bool:
        return self._is_trained


class PyTorchPredictor(BasePredictor):
    """
    PyTorch模型基类
    
    提供PyTorch模型的通用训练和预测功能
    针对GTX 1650优化 (4GB VRAM)
    """
    
    def __init__(
        self,
        name: str,
        device: Optional[str] = None,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        epochs: int = 100,
        early_stopping_patience: int = 10,
        **kwargs
    ):
        super().__init__(name, kwargs)
        
        # 自动选择设备
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # 针对GTX 1650优化的默认参数
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience
        
        self.model: Optional[nn.Module] = None
        self.optimizer = None
        self.criterion = None
        self.scheduler = None
        self.history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
        
        logger.info(f"使用设备: {self.device}")
    
    def _create_dataloader(
        self, 
        X: np.ndarray, 
        y: Optional[np.ndarray] = None, 
        shuffle: bool = True
    ) -> torch.utils.data.DataLoader:
        """创建DataLoader"""
        X_tensor = torch.FloatTensor(X)
        
        if y is not None:
            y_tensor = torch.LongTensor(y)
            dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        else:
            dataset = torch.utils.data.TensorDataset(X_tensor)
        
        return torch.utils.data.DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            shuffle=shuffle,
            pin_memory=True if self.device.type == 'cuda' else False
        )
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """训练模型"""
        if self.model is None:
            raise ValueError("请先调用build()构建模型")
        if self.optimizer is None or self.criterion is None:
            raise ValueError("请先调用build()构建优化器和损失函数")
        
        self.model.to(self.device)
        self.model.train()
        
        train_loader = self._create_dataloader(X_train, y_train, shuffle=True)
        val_loader = None
        if X_val is not None and y_val is not None:
            val_loader = self._create_dataloader(X_val, y_val, shuffle=False)
        
        best_val_loss = float('inf')
        patience_counter = 0
        best_state = None
        
        for epoch in range(self.epochs):
            # 训练阶段
            train_loss, train_correct = 0.0, 0
            self.model.train()
            
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                
                train_loss += loss.item() * len(batch_y)
                _, preds = torch.max(outputs, 1)
                train_correct += (preds == batch_y).sum().item()
            
            train_loss /= len(cast(Sized, train_loader.dataset))
            train_acc = train_correct / len(cast(Sized, train_loader.dataset))
            
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            
            # 验证阶段
            val_loss, val_acc = 0.0, 0.0
            if val_loader:
                val_loss, val_correct = 0.0, 0
                self.model.eval()
                
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        batch_X = batch_X.to(self.device)
                        batch_y = batch_y.to(self.device)
                        
                        outputs = self.model(batch_X)
                        loss = self.criterion(outputs, batch_y)
                        
                        val_loss += loss.item() * len(batch_y)
                        _, preds = torch.max(outputs, 1)
                        val_correct += (preds == batch_y).sum().item()
                
                val_loss /= len(cast(Sized, val_loader.dataset))
                val_acc = val_correct / len(cast(Sized, val_loader.dataset))
                
                self.history['val_loss'].append(val_loss)
                self.history['val_acc'].append(val_acc)
                
                # 早停检查
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_state = self.model.state_dict().copy()
                else:
                    patience_counter += 1
                
                if patience_counter >= self.early_stopping_patience:
                    logger.info(f"早停于第 {epoch+1} 轮")
                    break
            
            # 学习率调度
            if self.scheduler:
                self.scheduler.step(val_loss if val_loader else train_loss)
            
            # 日志
            if (epoch + 1) % 10 == 0 or epoch == 0:
                msg = f"Epoch {epoch+1}/{self.epochs}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}"
                if val_loader:
                    msg += f", val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
                logger.info(msg)
        
        # 恢复最佳模型
        if best_state:
            self.model.load_state_dict(best_state)
        
        self._is_trained = True
        return self.history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测类别"""
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """预测概率"""
        if not self._is_trained:
            raise ValueError("模型未训练")
        
        self.model.to(self.device)
        self.model.eval()
        
        dataloader = self._create_dataloader(X, shuffle=False)
        
        all_proba = []
        with torch.no_grad():
            for batch_X, in dataloader:
                batch_X = batch_X.to(self.device)
                outputs = self.model(batch_X)
                proba = torch.softmax(outputs, dim=1)
                all_proba.append(proba.cpu().numpy())
        
        return np.vstack(all_proba)
    
    def save(self, path: Path) -> None:
        """保存模型"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'history': self.history,
            'config': self.config
        }, path)
        logger.info(f"模型已保存: {path}")
    
    def load(self, path: Path) -> None:
        """加载模型"""
        path = Path(path)
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.history = checkpoint.get('history', {})
        self.config = checkpoint.get('config', {})
        self._is_trained = True
        
        logger.info(f"模型已加载: {path}")
