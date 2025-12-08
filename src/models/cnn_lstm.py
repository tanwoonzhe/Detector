"""
CNN-LSTM 混合模型
================================
使用CNN提取局部特征，LSTM捕获时序依赖

架构思路:
1. 1D CNN提取局部模式 (类似于技术分析中的短期形态)
2. LSTM捕获长期依赖
3. 结合两者的优势

参考:
- Kim, "Convolutional Neural Networks for Sentence Classification"
- CNN+LSTM时序预测模型 (Kaggle竞赛常见架构)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Tuple, Optional
import logging

from .base import PyTorchPredictor
from config import ModelConfig

logger = logging.getLogger(__name__)


class CNNLSTMNet(nn.Module):
    """
    CNN-LSTM混合网络
    
    架构:
    1. 多尺度1D CNN (不同kernel size捕获不同时间尺度的模式)
    2. LSTM编码器
    3. 全连接分类头
    """
    
    def __init__(
        self,
        input_size: int,
        cnn_filters: int = 64,
        kernel_sizes: list = [3, 5, 7],
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        n_classes: int = 3,
        dropout: float = 0.3
    ):
        super().__init__()
        
        self.kernel_sizes = kernel_sizes
        
        # 多尺度1D CNN
        # 每个CNN捕获不同时间尺度的局部模式
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(input_size, cnn_filters, kernel_size=k, padding=k//2),
                nn.BatchNorm1d(cnn_filters),
                nn.ReLU(),
                nn.Dropout(dropout * 0.5)
            )
            for k in kernel_sizes
        ])
        
        # CNN输出的通道数
        cnn_out_channels = cnn_filters * len(kernel_sizes)
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=cnn_out_channels,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0
        )
        
        # 层归一化
        self.layer_norm = nn.LayerNorm(lstm_hidden)
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden, lstm_hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden // 2, n_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向计算

        Args:
            x: (batch, seq_len, input_size)

        Returns:
            logits: (batch, n_classes)
        """
        # 转换维度用于1D CNN: (batch, input_size, seq_len)
        x_cnn = x.permute(0, 2, 1)
        
        # 多尺度CNN特征提取
        conv_outputs = []
        for conv in self.convs:
            conv_out = conv(x_cnn)  # (batch, cnn_filters, seq_len)
            conv_outputs.append(conv_out)
        
        # 拼接不同尺度的特征
        x_concat = torch.cat(conv_outputs, dim=1)  # (batch, cnn_filters * n_kernels, seq_len)
        
        # 转换维度用于LSTM: (batch, seq_len, cnn_filters * n_kernels)
        x_lstm = x_concat.permute(0, 2, 1)
        
        # LSTM编码
        lstm_out, (h_n, c_n) = self.lstm(x_lstm)
        
        # 取最后时刻的隐状态
        h_last = h_n[-1]  # (batch, lstm_hidden)
        
        # 层归一化
        h_norm = self.layer_norm(h_last)
        
        # 分类
        logits = self.classifier(h_norm)
        
        return logits


class ResidualCNNBlock(nn.Module):
    """残差CNN块"""
    
    def __init__(self, channels: int, kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        residual = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = self.bn2(self.conv2(x))
        x = x + residual
        x = self.relu(x)
        return x


class ResCNNLSTMNet(nn.Module):
    """
    残差CNN-LSTM网络
    
    使用残差连接增强CNN部分的特征提取能力
    """
    
    def __init__(
        self,
        input_size: int,
        cnn_filters: int = 64,
        n_res_blocks: int = 2,
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        n_classes: int = 3,
        dropout: float = 0.3
    ):
        super().__init__()
        
        # 输入投影
        self.input_proj = nn.Sequential(
            nn.Conv1d(input_size, cnn_filters, kernel_size=1),
            nn.BatchNorm1d(cnn_filters),
            nn.ReLU()
        )
        
        # 残差CNN块
        self.res_blocks = nn.Sequential(*[
            ResidualCNNBlock(cnn_filters, dropout=dropout * 0.5)
            for _ in range(n_res_blocks)
        ])
        
        # LSTM
        self.lstm = nn.LSTM(
            input_size=cnn_filters,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0
        )
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden, lstm_hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden // 2, n_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (batch, seq_len, input_size) -> (batch, input_size, seq_len)
        x = x.permute(0, 2, 1)
        
        # CNN特征提取
        x = self.input_proj(x)
        x = self.res_blocks(x)
        
        # (batch, cnn_filters, seq_len) -> (batch, seq_len, cnn_filters)
        x = x.permute(0, 2, 1)
        
        # LSTM编码
        _, (h_n, _) = self.lstm(x)
        h_last = h_n[-1]
        
        # 分类
        logits = self.classifier(h_last)
        
        return logits


class CNNLSTMPredictor(PyTorchPredictor):
    """
    CNN-LSTM预测器
    """
    
    def __init__(
        self,
        cnn_filters: int = 64,
        kernel_sizes: list = [3, 5, 7],
        lstm_hidden: Optional[int] = None,
        lstm_layers: Optional[int] = None,
        dropout: Optional[float] = None,
        use_residual: bool = False,
        **kwargs
    ):
        batch_size = kwargs.pop('batch_size', ModelConfig.BATCH_SIZE)
        learning_rate = kwargs.pop('learning_rate', ModelConfig.LEARNING_RATE)
        epochs = kwargs.pop('epochs', ModelConfig.EPOCHS)
        early_stopping_patience = kwargs.pop('early_stopping_patience', ModelConfig.EARLY_STOPPING_PATIENCE)

        super().__init__(
            name="CNN-LSTM" if not use_residual else "ResCNN-LSTM",
            batch_size=batch_size,
            learning_rate=learning_rate,
            epochs=epochs,
            early_stopping_patience=early_stopping_patience,
            **kwargs
        )
        
        self.cnn_filters = cnn_filters
        self.kernel_sizes = kernel_sizes
        self.lstm_hidden = lstm_hidden or ModelConfig.LSTM_HIDDEN_SIZE
        self.lstm_layers = lstm_layers or 2  # default LSTM layers
        self.dropout = dropout or 0.2  # default dropout
        self.use_residual = use_residual
    
    def build(self, input_shape: Tuple[int, ...], n_classes: int = 3) -> None:
        """构建模型"""
        seq_len, n_features = input_shape
        
        if self.use_residual:
            self.model = ResCNNLSTMNet(
                input_size=n_features,
                cnn_filters=self.cnn_filters,
                lstm_hidden=self.lstm_hidden,
                lstm_layers=self.lstm_layers,
                n_classes=n_classes,
                dropout=self.dropout
            )
        else:
            self.model = CNNLSTMNet(
                input_size=n_features,
                cnn_filters=self.cnn_filters,
                kernel_sizes=self.kernel_sizes,
                lstm_hidden=self.lstm_hidden,
                lstm_layers=self.lstm_layers,
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
        logger.info(f"{self.name}模型构建完成")
        logger.info(f"  输入形状: ({seq_len}, {n_features})")
        logger.info(f"  CNN过滤器: {self.cnn_filters}")
        logger.info(f"  LSTM隐藏层: {self.lstm_hidden}")
        logger.info(f"  总参数量: {total_params:,}")
