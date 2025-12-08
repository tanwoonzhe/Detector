"""
模型训练脚本
================================
训练BTC趋势预测模型

使用方法:
    python train.py --model gru --epochs 100
    python train.py --model all --epochs 50
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from config import ModelConfig, TradingConfig
from src.data_collection import CacheManager
from src.data_collection.coingecko_fetcher import CoinGeckoFetcher
from src.sentiment import SentimentAggregator
from src.features import FeatureEngineer
from src.validation import WalkForwardValidator, TimeSeriesMetrics
from src.models import (
    GRUPredictor, 
    BiLSTMPredictor, 
    CNNLSTMPredictor,
    LightGBMPredictor,
    ModelEnsemble
)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def fetch_data():
    """获取训练数据"""
    logger.info("获取历史数据...")
    
    # 价格数据
    fetcher = CoinGeckoFetcher()
    cache = CacheManager()
    
    market_data = await fetcher.get_hourly_ohlcv(
        symbol="bitcoin",
        vs_currency="usd",
        days=90  # 90天数据
    )
    
    # 转换为DataFrame
    df = market_data.to_dataframe()
    logger.info(f"获取到 {len(df)} 条价格数据")
    
    return df


def prepare_data(df: pd.DataFrame):
    """准备训练数据"""
    logger.info("特征工程...")
    
    engineer = FeatureEngineer()
    
    # 创建特征
    df_features = engineer.create_features(df)
    
    # 创建标签
    df_features = engineer.create_labels(df_features)
    
    # 准备训练数据
    X, y, feature_names = engineer.prepare_training_data(
        df_features, 
        target_window=1,  # 1小时预测
        for_classification=True
    )
    
    # 创建序列
    X_seq, y_seq = engineer.create_sequences(X, y)
    
    logger.info(f"特征维度: {X_seq.shape}")
    logger.info(f"类别分布: {np.bincount(y_seq.astype(int))}")
    
    return X_seq, y_seq, feature_names


def train_gru(X_train, y_train, X_val, y_val):
    """训练GRU模型"""
    logger.info("训练GRU模型...")
    
    model = GRUPredictor(
        hidden_size=ModelConfig.GRU_HIDDEN_SIZE,
        num_layers=ModelConfig.GRU_NUM_LAYERS,
        dropout=ModelConfig.DROPOUT,
        epochs=ModelConfig.EPOCHS,
        batch_size=ModelConfig.BATCH_SIZE,
        learning_rate=ModelConfig.LEARNING_RATE
    )
    
    model.build(input_shape=(X_train.shape[1], X_train.shape[2]), n_classes=3)
    history = model.train(X_train, y_train, X_val, y_val)
    
    return model, history


def train_bilstm(X_train, y_train, X_val, y_val):
    """训练BiLSTM模型"""
    logger.info("训练BiLSTM模型...")
    
    model = BiLSTMPredictor(
        hidden_size=ModelConfig.LSTM_HIDDEN_SIZE,
        num_layers=ModelConfig.LSTM_NUM_LAYERS,
        dropout=ModelConfig.DROPOUT,
        epochs=ModelConfig.EPOCHS,
        batch_size=ModelConfig.BATCH_SIZE
    )
    
    model.build(input_shape=(X_train.shape[1], X_train.shape[2]), n_classes=3)
    history = model.train(X_train, y_train, X_val, y_val)
    
    return model, history


def train_cnn_lstm(X_train, y_train, X_val, y_val):
    """训练CNN-LSTM模型"""
    logger.info("训练CNN-LSTM模型...")
    
    model = CNNLSTMPredictor(
        cnn_filters=64,
        kernel_sizes=[3, 5, 7],
        lstm_hidden=ModelConfig.LSTM_HIDDEN_SIZE,
        lstm_layers=2,
        dropout=ModelConfig.DROPOUT,
        epochs=ModelConfig.EPOCHS,
        batch_size=ModelConfig.BATCH_SIZE
    )
    
    model.build(input_shape=(X_train.shape[1], X_train.shape[2]), n_classes=3)
    history = model.train(X_train, y_train, X_val, y_val)
    
    return model, history


def train_lightgbm(X_train, y_train, X_val, y_val):
    """训练LightGBM模型"""
    logger.info("训练LightGBM模型...")
    
    model = LightGBMPredictor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05
    )
    
    model.build()
    history = model.train(X_train, y_train, X_val, y_val)
    
    return model, history


def evaluate_model(model, X_test, y_test, name: str):
    """评估模型"""
    logger.info(f"评估 {name}...")
    
    y_pred = model.predict(X_test)
    metrics = TimeSeriesMetrics.calculate_metrics(y_test, y_pred)
    
    logger.info(f"  准确率: {metrics['accuracy']:.4f}")
    logger.info(f"  F1分数: {metrics['f1_macro']:.4f}")
    
    return metrics


def walk_forward_validation(X, y, model_class, model_kwargs):
    """Walk-Forward验证"""
    logger.info("执行Walk-Forward验证...")
    
    validator = WalkForwardValidator(
        train_size=168 * 4,  # 4周训练
        test_size=168,       # 1周测试
        step_size=24,        # 每天滚动
        expanding=True
    )
    
    all_metrics = []
    
    for fold, (train_idx, test_idx) in enumerate(validator.split(X)):
        logger.info(f"  Fold {fold + 1}...")
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # 训练模型
        model = model_class(**model_kwargs)
        model.build(input_shape=(X_train.shape[1], X_train.shape[2]), n_classes=3)
        model.train(X_train, y_train)
        
        # 评估
        y_pred = model.predict(X_test)
        metrics = TimeSeriesMetrics.calculate_metrics(y_test, y_pred)
        all_metrics.append(metrics)
        
        if fold >= 4:  # 限制fold数量
            break
    
    # 平均指标
    avg_metrics = {
        key: np.mean([m[key] for m in all_metrics])
        for key in all_metrics[0].keys()
    }
    
    logger.info(f"  平均准确率: {avg_metrics['accuracy']:.4f}")
    logger.info(f"  平均F1: {avg_metrics['f1_macro']:.4f}")
    
    return avg_metrics


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='训练BTC趋势预测模型')
    parser.add_argument('--model', type=str, default='gru',
                       choices=['gru', 'bilstm', 'cnn_lstm', 'lightgbm', 'all'],
                       help='要训练的模型')
    parser.add_argument('--epochs', type=int, default=ModelConfig.EPOCHS,
                       help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=ModelConfig.BATCH_SIZE,
                       help='批次大小')
    parser.add_argument('--validate', action='store_true',
                       help='是否执行Walk-Forward验证')
    args = parser.parse_args()
    
    # 更新配置
    ModelConfig.EPOCHS = args.epochs
    ModelConfig.BATCH_SIZE = args.batch_size
    
    logger.info("="*50)
    logger.info("BTC趋势预测模型训练")
    logger.info("="*50)
    
    # 获取数据
    try:
        df = asyncio.run(fetch_data())
    except Exception as e:
        logger.error(f"获取数据失败: {e}")
        logger.info("使用模拟数据进行演示...")
        
        # 生成模拟数据
        np.random.seed(42)
        dates = pd.date_range(end=datetime.now(), periods=2160, freq='H')
        returns = np.random.randn(2160) * 0.01
        prices = 65000 * np.cumprod(1 + returns)
        
        df = pd.DataFrame({
            'open': prices * (1 - np.random.rand(2160) * 0.005),
            'high': prices * (1 + np.random.rand(2160) * 0.01),
            'low': prices * (1 - np.random.rand(2160) * 0.01),
            'close': prices,
            'volume': np.random.rand(2160) * 1e9
        }, index=dates)
    
    # 准备数据
    X_seq, y_seq, feature_names = prepare_data(df)
    
    # 分割训练/验证/测试集
    n = len(X_seq)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)
    
    X_train = X_seq[:train_end]
    y_train = y_seq[:train_end]
    X_val = X_seq[train_end:val_end]
    y_val = y_seq[train_end:val_end]
    X_test = X_seq[val_end:]
    y_test = y_seq[val_end:]
    
    logger.info(f"训练集: {len(X_train)}, 验证集: {len(X_val)}, 测试集: {len(X_test)}")
    
    # 保存模型的目录
    model_dir = Path(__file__).parent / 'models' / 'saved'
    model_dir.mkdir(parents=True, exist_ok=True)
    
    models = {}
    
    # 训练模型
    if args.model in ['gru', 'all']:
        model, _ = train_gru(X_train, y_train, X_val, y_val)
        models['gru'] = model
        evaluate_model(model, X_test, y_test, 'GRU')
        model.save(model_dir / 'gru_model.pt')
    
    if args.model in ['bilstm', 'all']:
        model, _ = train_bilstm(X_train, y_train, X_val, y_val)
        models['bilstm'] = model
        evaluate_model(model, X_test, y_test, 'BiLSTM')
        model.save(model_dir / 'bilstm_model.pt')
    
    if args.model in ['cnn_lstm', 'all']:
        model, _ = train_cnn_lstm(X_train, y_train, X_val, y_val)
        models['cnn_lstm'] = model
        evaluate_model(model, X_test, y_test, 'CNN-LSTM')
        model.save(model_dir / 'cnn_lstm_model.pt')
    
    if args.model in ['lightgbm', 'all']:
        model, _ = train_lightgbm(X_train, y_train, X_val, y_val)
        models['lightgbm'] = model
        evaluate_model(model, X_test, y_test, 'LightGBM')
        model.save(model_dir / 'lightgbm_model.pkl')
    
    # 集成模型
    if args.model == 'all' and len(models) > 1:
        logger.info("创建集成模型...")
        ensemble = ModelEnsemble(
            models=list(models.values()),
            strategy='soft_voting'
        )
        
        # 评估集成
        y_pred = ensemble.predict(X_test)
        metrics = TimeSeriesMetrics.calculate_metrics(y_test, y_pred)
        logger.info(f"集成准确率: {metrics['accuracy']:.4f}")
        logger.info(f"集成F1: {metrics['f1_macro']:.4f}")
    
    # Walk-Forward验证
    if args.validate:
        logger.info("\n执行Walk-Forward验证...")
        wf_metrics = walk_forward_validation(
            X_seq, y_seq,
            GRUPredictor,
            {'hidden_size': 128, 'num_layers': 2, 'epochs': 50}
        )
    
    logger.info("\n" + "="*50)
    logger.info("训练完成！")
    logger.info(f"模型已保存到: {model_dir}")
    logger.info("="*50)


if __name__ == "__main__":
    main()
