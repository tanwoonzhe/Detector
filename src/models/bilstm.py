"""
BiLSTM 双向LSTM模型
================================
使用双向LSTM捕获时序数据的前后文信息

参考:
- Hochreiter & Schmidhuber, "Long Short-Term Memory"
- Graves et al., "Framewise Phoneme Classification with Bidirectional LSTM"
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Tuple, Optional
import logging

from .base import PyTorchPredictor
from config import ModelConfig

logger = logging.getLogger(__name__)


class BiLSTMNet(nn.Module):
    """
    双向LSTM网络
    
    架构:
    1. 双向LSTM编码器
    2. 前向和后向隐状态拼接
    3. 全连接分类头
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        n_classes: int = 3,
        dropout: float = 0.3
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # 双向LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # 层归一化
        self.layer_norm = nn.LayerNorm(hidden_size * 2)
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, n_classes)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
                # LSTM forget gate bias设为1
                if 'bias_ih' in name or 'bias_hh' in name:
                    n = param.size(0)
                    param.data[n//4:n//2].fill_(1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_size)
            
        Returns:
            logits: (batch, n_classes)
        """
        # LSTM编码
        # lstm_out: (batch, seq_len, hidden_size * 2)
        # h_n: (num_layers * 2, batch, hidden_size)
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # 拼接最后时刻的前向和后向隐状态
        # 前向最后隐状态: h_n[-2]
        # 后向最后隐状态: h_n[-1]
        h_forward = h_n[-2]  # (batch, hidden_size)
        h_backward = h_n[-1]  # (batch, hidden_size)
        
        # 拼接
        h_concat = torch.cat([h_forward, h_backward], dim=1)  # (batch, hidden_size * 2)
        
        # 层归一化
        h_norm = self.layer_norm(h_concat)
        
        # 分类
        logits = self.classifier(h_norm)
        
        return logits


class BiLSTMPredictor(PyTorchPredictor):
    """
    BiLSTM预测器
    """
    
    def __init__(
        self,
        hidden_size: Optional[int] = None,
        num_layers: Optional[int] = None,
        dropout: Optional[float] = None,
        **kwargs
    ):
        batch_size = kwargs.pop('batch_size', ModelConfig.BATCH_SIZE)
        learning_rate = kwargs.pop('learning_rate', ModelConfig.LEARNING_RATE)
        epochs = kwargs.pop('epochs', ModelConfig.EPOCHS)
        early_stopping_patience = kwargs.pop('early_stopping_patience', ModelConfig.EARLY_STOPPING_PATIENCE)

        super().__init__(
            name="BiLSTM",
            batch_size=batch_size,
            learning_rate=learning_rate,
            epochs=epochs,
            early_stopping_patience=early_stopping_patience,
            **kwargs
        )
        
        self.hidden_size = hidden_size or ModelConfig.BILSTM_HIDDEN_SIZE
        self.num_layers = num_layers or ModelConfig.BILSTM_NUM_LAYERS
        self.dropout = dropout or ModelConfig.BILSTM_DROPOUT
    
    def build(self, input_shape: Tuple[int, ...], n_classes: int = 3) -> None:
        """构建模型"""
        seq_len, n_features = input_shape
        self.input_shape = input_shape  # 保存输入形状
        self.n_classes = n_classes
        
        # 保存到config供save/load使用
        if not isinstance(self.config, dict):
            self.config = {}
        self.config['n_classes'] = n_classes
        self.config['hidden_size'] = self.hidden_size
        self.config['num_layers'] = self.num_layers
        self.config['dropout'] = self.dropout
        
        self.model = BiLSTMNet(
            input_size=n_features,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            n_classes=n_classes,
            dropout=self.dropout
        )
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-5
        )
        
        self.criterion = nn.CrossEntropyLoss()
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"BiLSTM模型构建完成")
        logger.info(f"  输入形状: ({seq_len}, {n_features})")
        logger.info(f"  隐藏层大小: {self.hidden_size}")
        logger.info(f"  LSTM层数: {self.num_layers}")
        logger.info(f"  总参数量: {total_params:,}")
    
    def load(self, path: Path, auto_build: bool = True) -> None:
        """
        加载模型 - 覆盖基类方法以正确恢复超参数
        
        Args:
            path: 模型文件路径
            auto_build: 如果模型未构建，是否自动从checkpoint中读取配置并构建
        """
        path = Path(path)
        checkpoint = torch.load(path, map_location=self.device)
        
        # 从checkpoint恢复超参数
        if 'config' in checkpoint:
            config = checkpoint['config']
            if 'hidden_size' in config:
                self.hidden_size = config['hidden_size']
            if 'num_layers' in config:
                self.num_layers = config['num_layers']
            if 'dropout' in config:
                self.dropout = config['dropout']
        
        # 如果模型未构建，尝试自动构建
        if self.model is None:
            if auto_build and 'config' in checkpoint and 'input_shape' in checkpoint['config']:
                input_shape = checkpoint['config']['input_shape']
                n_classes = checkpoint['config'].get('n_classes', 3)
                logger.info(f"从checkpoint自动构建BiLSTM模型:")
                logger.info(f"  input_shape={input_shape}, n_classes={n_classes}")
                logger.info(f"  hidden_size={self.hidden_size}, num_layers={self.num_layers}")
                self.build(input_shape=tuple(input_shape), n_classes=n_classes)
            else:
                raise RuntimeError("模型未构建！请先调用 build() 方法，或在checkpoint中包含input_shape信息")
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.history = checkpoint.get('history', {})
        self.config = checkpoint.get('config', {})
        self._is_trained = True
        
        logger.info(f"BiLSTM模型已加载: {path}")
