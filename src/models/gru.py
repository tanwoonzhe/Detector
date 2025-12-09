"""
GRU + Attention 模型
================================
基于GRU的序列预测模型，带自注意力机制
针对GTX 1650 (4GB VRAM) 优化

参考:
- Cho et al., "Learning Phrase Representations using RNN Encoder-Decoder"
- Vaswani et al., "Attention Is All You Need"
- CSDN上的LSTM股票预测实现
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Tuple, Optional
from pathlib import Path
import logging

from .base import PyTorchPredictor
from config import ModelConfig

logger = logging.getLogger(__name__)


class Attention(nn.Module):
    """
    简单的自注意力层
    
    计算序列中每个时间步的重要性权重
    """
    
    def __init__(self, hidden_size: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1)
        )
    
    def forward(self, gru_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            gru_output: (batch, seq_len, hidden_size)
            
        Returns:
            context: (batch, hidden_size) - 加权上下文向量
            weights: (batch, seq_len) - 注意力权重
        """
        # (batch, seq_len, 1)
        attention_scores = self.attention(gru_output)
        
        # (batch, seq_len)
        attention_weights = F.softmax(attention_scores.squeeze(-1), dim=1)
        
        # (batch, hidden_size) = sum over seq_len
        context = torch.bmm(attention_weights.unsqueeze(1), gru_output).squeeze(1)
        
        return context, attention_weights


class GRUAttentionNet(nn.Module):
    """
    GRU + Attention 网络
    
    架构:
    1. 多层GRU编码器
    2. 自注意力层
    3. 全连接分类头
    
    针对GTX 1650优化:
    - 使用2层GRU (而非3层)
    - hidden_size=128 (而非256)
    - 适当的dropout防止过拟合
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        n_classes: int = 3,
        dropout: float = 0.3,
        bidirectional: bool = False
    ):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.n_directions = 2 if bidirectional else 1
        
        # GRU层
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # 注意力层
        self.attention = Attention(hidden_size * self.n_directions)
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * self.n_directions, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, n_classes)
        )
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """Xavier初始化"""
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_size)
            
        Returns:
            logits: (batch, n_classes)
        """
        # GRU编码
        # gru_out: (batch, seq_len, hidden_size * n_directions)
        gru_out, _ = self.gru(x)
        
        # 注意力聚合
        # context: (batch, hidden_size * n_directions)
        context, attention_weights = self.attention(gru_out)
        
        # 分类
        logits = self.classifier(context)
        
        return logits
    
    def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """获取注意力权重 (用于可视化)"""
        gru_out, _ = self.gru(x)
        _, attention_weights = self.attention(gru_out)
        return attention_weights


class GRUPredictor(PyTorchPredictor):
    """
    GRU预测器
    
    封装GRU模型的训练和预测
    """
    
    def __init__(
        self,
        hidden_size: Optional[int] = None,
        num_layers: Optional[int] = None,
        dropout: Optional[float] = None,
        bidirectional: bool = False,
        **kwargs
    ):
        # 取出会传递给父类的关键参数，避免重复传参
        batch_size = kwargs.pop('batch_size', ModelConfig.BATCH_SIZE)
        learning_rate = kwargs.pop('learning_rate', ModelConfig.LEARNING_RATE)
        epochs = kwargs.pop('epochs', ModelConfig.EPOCHS)
        early_stopping_patience = kwargs.pop('early_stopping_patience', ModelConfig.EARLY_STOPPING_PATIENCE)

        super().__init__(
            name="GRU-Attention",
            batch_size=batch_size,
            learning_rate=learning_rate,
            epochs=epochs,
            early_stopping_patience=early_stopping_patience,
            **kwargs
        )
        
        # 模型参数
        self.hidden_size = hidden_size or ModelConfig.GRU_HIDDEN_SIZE
        self.num_layers = num_layers or ModelConfig.GRU_NUM_LAYERS
        self.dropout = dropout or ModelConfig.GRU_DROPOUT
        self.bidirectional = bidirectional
        
        self.n_classes = 3  # 下跌, 横盘, 上涨
    
    def build(self, input_shape: Tuple[int, ...], n_classes: int = 3) -> None:
        """
        构建模型
        
        Args:
            input_shape: (seq_len, n_features)
            n_classes: 类别数
        """
        seq_len, n_features = input_shape
        self.n_classes = n_classes
        self.input_shape = input_shape  # 保存输入形状
        
        self.model = GRUAttentionNet(
            input_size=n_features,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            n_classes=n_classes,
            dropout=self.dropout,
            bidirectional=self.bidirectional
        )
        
        # 优化器
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-5
        )
        
        # 损失函数 (类别加权)
        self.criterion = nn.CrossEntropyLoss()
        
        # 学习率调度
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5
        )
        
        # 打印模型信息
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        logger.info(f"GRU模型构建完成")
        logger.info(f"  输入形状: ({seq_len}, {n_features})")
        logger.info(f"  隐藏层大小: {self.hidden_size}")
        logger.info(f"  GRU层数: {self.num_layers}")
        logger.info(f"  双向: {self.bidirectional}")
        logger.info(f"  总参数量: {total_params:,}")
        logger.info(f"  可训练参数: {trainable_params:,}")
    
    def get_attention_weights(self, X: np.ndarray) -> np.ndarray:
        """
        获取注意力权重
        
        Args:
            X: (batch, seq_len, features)
            
        Returns:
            weights: (batch, seq_len)
        """
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        with torch.no_grad():
            weights = self.model.get_attention_weights(X_tensor)
        
        return weights.cpu().numpy()


class MultiWindowGRU(nn.Module):
    """
    多窗口GRU
    
    同时预测多个时间窗口 (0.5h, 1h, 2h, 4h)
    共享GRU编码器，每个窗口有独立的分类头
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        n_classes: int = 3,
        prediction_windows: list = [0.5, 1, 2, 4],
        dropout: float = 0.3
    ):
        super().__init__()
        
        self.prediction_windows = prediction_windows
        
        # 共享GRU编码器
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 共享注意力
        self.attention = Attention(hidden_size)
        
        # 每个窗口的独立分类头
        self.classifiers = nn.ModuleDict()
        for window in prediction_windows:
            window_key = str(window).replace('.', '_')
            self.classifiers[window_key] = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size // 2, n_classes)
            )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, input_size)
            
        Returns:
            outputs: {window: (batch, n_classes)}
        """
        # 共享编码
        gru_out, _ = self.gru(x)
        context, _ = self.attention(gru_out)
        
        # 各窗口预测
        outputs = {}
        for window in self.prediction_windows:
            window_key = str(window).replace('.', '_')
            outputs[window] = self.classifiers[window_key](context)
        
        return outputs


class MultiWindowGRUPredictor(PyTorchPredictor):
    """
    多窗口GRU预测器
    """
    
    def __init__(self, prediction_windows: list = [0.5, 1, 2, 4], **kwargs):
        super().__init__(name="MultiWindow-GRU", **kwargs)
        self.prediction_windows = prediction_windows
        self.hidden_size = kwargs.get('hidden_size', ModelConfig.GRU_HIDDEN_SIZE)
        self.num_layers = kwargs.get('num_layers', ModelConfig.GRU_NUM_LAYERS)
        self.dropout = kwargs.get('dropout', ModelConfig.GRU_DROPOUT)
    
    def build(self, input_shape: Tuple[int, ...], n_classes: int = 3) -> None:
        seq_len, n_features = input_shape
        
        self.model = MultiWindowGRU(
            input_size=n_features,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            n_classes=n_classes,
            prediction_windows=self.prediction_windows,
            dropout=self.dropout
        )
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-5
        )
        
        # 多任务损失 (各窗口的平均)
        self.criterion = nn.CrossEntropyLoss()
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        
        logger.info(f"多窗口GRU模型构建完成: windows={self.prediction_windows}")
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: Dict[float, np.ndarray],
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[Dict[float, np.ndarray]] = None
    ) -> Dict[str, Any]:
        """
        训练多窗口模型
        
        Args:
            X_train: (samples, seq_len, features)
            y_train: {window: labels}
        """
        self.model.to(self.device)
        
        # 构造DataLoader
        X_tensor = torch.FloatTensor(X_train)
        y_tensors = {w: torch.LongTensor(y) for w, y in y_train.items()}
        
        dataset = torch.utils.data.TensorDataset(
            X_tensor, 
            *[y_tensors[w] for w in self.prediction_windows]
        )
        train_loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True
        )
        
        best_val_loss = float('inf')
        patience_counter = 0
        best_state = None
        
        for epoch in range(self.epochs):
            self.model.train()
            train_loss = 0.0
            
            for batch in train_loader:
                X_batch = batch[0].to(self.device)
                y_batches = {
                    w: batch[i+1].to(self.device) 
                    for i, w in enumerate(self.prediction_windows)
                }
                
                self.optimizer.zero_grad()
                outputs = self.model(X_batch)
                
                # 多任务损失
                loss = torch.tensor(0.0, device=self.device)
                for w in self.prediction_windows:
                    loss = loss + self.criterion(outputs[w], y_batches[w])
                loss = loss / len(self.prediction_windows)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                train_loss += loss.item() * len(X_batch)
            
            train_loss /= len(train_loader.dataset)
            self.history['train_loss'].append(train_loss)
            
            # 验证
            if X_val is not None and y_val is not None:
                val_loss = self._validate(X_val, y_val)
                self.history['val_loss'].append(val_loss)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_state = self.model.state_dict().copy()
                else:
                    patience_counter += 1
                
                if patience_counter >= self.early_stopping_patience:
                    logger.info(f"早停于第 {epoch+1} 轮")
                    break
                
                self.scheduler.step(val_loss)
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}: train_loss={train_loss:.4f}")
        
        if best_state:
            self.model.load_state_dict(best_state)
        
        self._is_trained = True
        return self.history
    
    def _validate(self, X_val: np.ndarray, y_val: Dict[float, np.ndarray]) -> float:
        """验证"""
        self.model.eval()
        X_tensor = torch.FloatTensor(X_val).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(X_tensor)
            
            loss = torch.tensor(0.0, device=self.device)
            for w in self.prediction_windows:
                y_tensor = torch.LongTensor(y_val[w]).to(self.device)
                loss = loss + self.criterion(outputs[w], y_tensor)
            loss = loss / len(self.prediction_windows)
        
        return loss.item()
    
    def predict(self, X: np.ndarray, window: float = 1.0) -> np.ndarray:
        """预测指定窗口"""
        proba = self.predict_proba(X, window)
        return np.argmax(proba, axis=1)
    
    def predict_proba(self, X: np.ndarray, window: float = 1.0) -> np.ndarray:
        """预测指定窗口的概率"""
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(X_tensor)
            proba = torch.softmax(outputs[window], dim=1)
        
        return proba.cpu().numpy()
    
    def predict_all_windows(self, X: np.ndarray) -> Dict[float, np.ndarray]:
        """预测所有窗口"""
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(X_tensor)
            results = {}
            for w in self.prediction_windows:
                proba = torch.softmax(outputs[w], dim=1)
                results[w] = np.argmax(proba.cpu().numpy(), axis=1)
        
        return results
