"""
BTCUSDT 小时级AI预测系统
================================
主要模块:
- data_collection: 数据采集（CoinGecko/Binance）
- sentiment: 情感分析（CryptoBERT/VADER）
- features: 特征工程（技术指标/蜡烛图形态）
- models: 预测模型（GRU/BiLSTM/CNN-LSTM/LightGBM）
- validation: 验证框架（Purged K-Fold/Walk-Forward）
- signals: 信号生成器
"""

__version__ = "1.0.0"
__author__ = "BTCUSDT Predictor Team"
