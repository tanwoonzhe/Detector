"""
模型训练脚本
================================
训练BTC趋势预测模型

使用方法:
    python train.py --model gru --epochs 100
    python train.py --model all --epochs 50
    python train.py --model cnn_lstm --use-hf-multi --interval 15min --epochs 100
    python train.py --use-binance-hist --interval 5min --days 365
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

from config import ModelConfig, TradingConfig, FeatureConfig, APIConfig
from src.data_collection import CacheManager
from src.data_collection.coingecko_fetcher import CoinGeckoFetcher
from src.data_collection.fmp_fetcher import FMPFetcher
from src.data_collection.data_pipeline import DataPipeline
from src.data_collection.binance_historical import BinanceHistoricalFetcher, download_btc_historical, load_btc_historical
from src.data_collection.fred_fetcher import FREDFetcher
from src.data_collection.kaggle_fetcher import KaggleFetcher, load_kaggle_btc
from src.data_collection.hf_loader_multi import load_hf_btc_multi_granularity
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


async def fetch_data(
    use_hf: bool = False, 
    merge_recent: bool = False, 
    use_fmp: bool = False, 
    fmp_days: int = 90,
    use_pipeline: bool = False,
    include_macro: bool = True,
    include_onchain: bool = True,
    use_hf_multi: bool = False,
    interval: str = "1h",
    use_binance_hist: bool = False,
    use_kaggle: bool = False,
    days: int = 90
):
    """
    获取训练数据
    
    Args:
        use_hf: 使用HuggingFace历史数据集（小时级）
        merge_recent: 合并最近的CoinGecko数据（与use_hf一起使用）
        use_fmp: 使用Financial Modeling Prep (FMP) API
        fmp_days: FMP数据天数
        use_pipeline: 使用多数据源管道（宏观+链上+跨市场）
        include_macro: 包含宏观数据（需要FMP API）
        include_onchain: 包含链上数据（CoinMetrics）
        use_hf_multi: 使用多粒度HuggingFace数据
        interval: 数据间隔 (1min, 5min, 15min, 30min, 1h, 4h, 1d)
        use_binance_hist: 使用Binance历史归档数据
        use_kaggle: 使用Kaggle历史数据
        days: 获取数据的天数
    """
    logger.info("获取历史数据...")
    
    df = None
    
    # 选项A: 使用Binance历史归档数据（高优先级，最完整）
    if use_binance_hist:
        logger.info(f"📥 使用 Binance 历史归档数据 ({interval})...")
        try:
            # 先尝试加载本地缓存
            df = load_btc_historical(interval=interval)
            
            if df is None or df.empty:
                # 如果没有缓存，下载数据
                logger.info("本地无缓存，开始下载Binance历史数据...")
                df = asyncio.get_event_loop().run_until_complete(
                    download_btc_historical(interval=interval)
                )
            
            if df is not None and not df.empty:
                logger.info(f"✅ Binance历史数据加载成功: {len(df)} 条记录")
                logger.info(f"   时间范围: {df.index.min()} ~ {df.index.max()}")
                logger.info(f"   数据间隔: {interval}")
            else:
                logger.warning("⚠️ Binance历史数据为空，回退到其他数据源")
                df = None
                
        except Exception as e:
            logger.error(f"❌ Binance历史数据加载异常: {e}")
            logger.info("回退到其他数据源...")
            df = None
    
    # 选项B: 使用多粒度HuggingFace数据
    if df is None and use_hf_multi:
        logger.info(f"📥 加载多粒度 HuggingFace 数据集 ({interval})...")
        try:
            df = load_hf_btc_multi_granularity(granularity=interval)
            
            if df is not None and not df.empty:
                logger.info(f"✅ HF多粒度数据加载成功: {len(df)} 条记录")
                logger.info(f"   时间范围: {df.index.min()} ~ {df.index.max()}")
                logger.info(f"   数据间隔: {interval}")
            else:
                logger.warning("⚠️ HF多粒度数据为空，回退到其他数据源")
                df = None
                
        except Exception as e:
            logger.error(f"❌ HF多粒度数据加载异常: {e}")
            logger.info("回退到其他数据源...")
            df = None
    
    # 选项C: 使用Kaggle数据
    if df is None and use_kaggle:
        logger.info("📥 加载 Kaggle 历史数据...")
        try:
            resample_to = interval if interval in ["1min", "1h", "1d"] else "1h"
            df = load_kaggle_btc(resample_to=resample_to)
            
            if df is not None and not df.empty:
                logger.info(f"✅ Kaggle数据加载成功: {len(df)} 条记录")
                logger.info(f"   时间范围: {df.index.min()} ~ {df.index.max()}")
            else:
                logger.warning("⚠️ Kaggle数据为空，回退到其他数据源")
                df = None
                
        except Exception as e:
            logger.error(f"❌ Kaggle数据加载异常: {e}")
            logger.info("回退到其他数据源...")
            df = None
    
    # 选项0: 使用多数据源管道
    if use_pipeline:
        logger.info("📥 使用多数据源管道获取数据...")
        try:
            pipeline = DataPipeline(
                fmp_api_key=APIConfig.FMP_API_KEY,
                coinmetrics_api_key=getattr(APIConfig, 'COINMETRICS_API_KEY', '')
            )
            
            df = await pipeline.fetch_all(
                days=fmp_days,
                include_macro=include_macro and bool(APIConfig.FMP_API_KEY),
                include_onchain=include_onchain,
                include_cross_asset=True,
                resample_to_hourly=True
            )
            
            await pipeline.close()
            
            if not df.empty:
                logger.info(f"✅ 多源数据加载成功: {len(df)} 条记录, {len(df.columns)} 列")
                logger.info(f"   时间范围: {df.index.min()} ~ {df.index.max()}")
                return df
            else:
                logger.warning("⚠️ 多源数据为空，回退到其他数据源")
                df = None
                
        except Exception as e:
            logger.error(f"❌ 多源数据加载异常: {e}")
            logger.info("回退到其他数据源...")
            df = None
    
    # 选项1: 使用 FMP 数据
    if df is None and use_fmp:
        logger.info("📥 使用 FMP API 获取数据...")
        try:
            api_key = APIConfig.FMP_API_KEY
            if not api_key:
                logger.warning("⚠️ FMP_API_KEY 未设置，请在 .env 文件中配置")
                raise ValueError("FMP API密钥未设置")
            
            fetcher = FMPFetcher(api_key=api_key)
            market_data = await fetcher.get_hourly_ohlcv(
                symbol="BTCUSD",
                days=fmp_days
            )
            await fetcher.close()
            
            df = market_data.to_dataframe()
            
            if not df.empty:
                logger.info(f"✅ FMP数据加载成功: {len(df)} 条记录")
                logger.info(f"   时间范围: {df.index.min()} ~ {df.index.max()}")
            else:
                logger.warning("⚠️ FMP数据为空，回退到其他数据源")
                df = None
                
        except Exception as e:
            logger.error(f"❌ FMP数据加载异常: {e}")
            logger.info("回退到其他数据源...")
            df = None
    
    # 选项2: 使用 HuggingFace 历史数据
    if df is None and use_hf:
        logger.info("📥 加载 HuggingFace 历史数据集...")
        try:
            from src.data_collection.hf_loader_fixed import load_hf_btc_data
            df = load_hf_btc_data()
            
            if not df.empty:
                logger.info(f"✅ HF数据加载成功: {len(df)} 条记录")
                logger.info(f"   时间范围: {df.index.min()} ~ {df.index.max()}")
                
                # 如果需要合并最新数据
                if merge_recent:
                    logger.info("📊 合并最新 CoinGecko 数据...")
                    fetcher = CoinGeckoFetcher()
                    recent_data = await fetcher.get_hourly_ohlcv(
                        symbol="bitcoin",
                        vs_currency="usd",
                        days=7  # 获取最近7天数据
                    )
                    await fetcher.close()
                    
                    df_recent = recent_data.to_dataframe()
                    
                    # 统一时区处理：移除时区信息进行比较
                    df_max_time = df.index.max()
                    if isinstance(df.index, pd.DatetimeIndex):
                        if df.index.tz is not None:
                            df.index = df.index.tz_localize(None)  # type: ignore
                    if isinstance(df_recent.index, pd.DatetimeIndex):
                        if df_recent.index.tz is not None:
                            df_recent.index = df_recent.index.tz_localize(None)  # type: ignore
                    if isinstance(df_max_time, pd.Timestamp) and df_max_time.tz is not None:
                        df_max_time = df_max_time.tz_localize(None)
                    
                    # 只保留 HF 数据之后的部分
                    df_recent = df_recent[df_recent.index > df_max_time]
                    
                    if not df_recent.empty:
                        logger.info(f"   新增 {len(df_recent)} 条最新数据")
                        df = pd.concat([df, df_recent]).sort_index()
                    
            else:
                logger.warning("⚠️ HF数据加载失败，回退到 CoinGecko")
                df = None
                
        except Exception as e:
            logger.error(f"❌ HF数据加载异常: {e}")
            df = None
    
    # 如果没有使用HF或HF加载失败，使用 CoinGecko
    if df is None or df.empty:
        logger.info("📊 使用 CoinGecko 获取90天小时数据...")
        fetcher = CoinGeckoFetcher()
        
        market_data = await fetcher.get_hourly_ohlcv(
            symbol="bitcoin",
            vs_currency="usd",
            days=90  # 90天数据，约2160条
        )
        
        # 转换为DataFrame
        df = market_data.to_dataframe()
        logger.info(f"原始数据: {len(df)} 条 (范围: {df.index.min()} ~ {df.index.max()})")
        
        await fetcher.close()
    
    # 确保时区一致
    if hasattr(df.index, 'tz'):
        if df.index.tz is None:  # type: ignore
            df.index = df.index.tz_localize('UTC')  # type: ignore
        else:
            df.index = df.index.tz_convert('UTC')  # type: ignore
    
    return df


def prepare_data(df: pd.DataFrame):
    """准备训练数据"""
    logger.info(f"特征工程开始... 初始数据: {len(df)} 行")
    
    engineer = FeatureEngineer()
    
    # 创建特征
    df_features = engineer.create_features(df)
    logger.info(f"特征创建后: {len(df_features)} 行")
    
    if len(df_features) < 50:
        logger.error(f"特征工程后数据不足: {len(df_features)} 行 < 50 行最小要求")
        raise ValueError(
            f"特征工程后仅剩 {len(df_features)} 行数据，不足以训练。"
            f"建议：1) 使用更多天数的数据 2) 减小特征窗口大小（当前SEQUENCE_LENGTH={ModelConfig.SEQUENCE_LENGTH}）"
        )
    
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
    
    # === 传统数据源 ===
    parser.add_argument('--use-hf', action='store_true',
                       help='使用HuggingFace历史数据集（小时级）')
    parser.add_argument('--merge-recent', action='store_true',
                       help='合并最近的CoinGecko数据（与--use-hf一起使用）')
    parser.add_argument('--use-fmp', action='store_true',
                       help='使用Financial Modeling Prep (FMP) API获取数据')
    parser.add_argument('--fmp-days', type=int, default=90,
                       help='数据天数（默认90天）')
    
    # === 新增长历史数据源 ===
    parser.add_argument('--use-hf-multi', action='store_true',
                       help='使用多粒度HuggingFace数据（支持1min/5min/15min/30min/1h/4h/1d）')
    parser.add_argument('--use-binance-hist', action='store_true',
                       help='使用Binance历史归档数据（2017至今，官方数据）')
    parser.add_argument('--use-kaggle', action='store_true',
                       help='使用Kaggle BTC历史数据（2012至今）')
    parser.add_argument('--interval', type=str, default='1h',
                       choices=['1min', '5min', '15min', '30min', '1h', '4h', '1d'],
                       help='数据间隔/粒度（默认1h）')
    parser.add_argument('--days', type=int, default=365,
                       help='获取数据天数（默认365天）')
    
    # === 多数据源管道参数 ===
    parser.add_argument('--use-pipeline', action='store_true',
                       help='使用多数据源管道（合并宏观+链上+跨市场数据）')
    parser.add_argument('--include-macro', action='store_true', default=True,
                       help='包含宏观经济数据（需要FMP API）')
    parser.add_argument('--include-onchain', action='store_true', default=True,
                       help='包含链上数据（CoinMetrics）')
    parser.add_argument('--no-macro', action='store_true',
                       help='不包含宏观数据')
    parser.add_argument('--no-onchain', action='store_true',
                       help='不包含链上数据')
    
    args = parser.parse_args()
    
    # 处理参数
    include_macro = not args.no_macro
    include_onchain = not args.no_onchain
    
    # 更新配置
    ModelConfig.EPOCHS = args.epochs
    ModelConfig.BATCH_SIZE = args.batch_size
    
    logger.info("="*50)
    logger.info("BTC趋势预测模型训练")
    logger.info("="*50)
    
    # 显示数据源选择
    if args.use_binance_hist:
        logger.info(f"📊 数据源: Binance历史归档 (间隔: {args.interval}, {args.days}天)")
    elif args.use_hf_multi:
        logger.info(f"📊 数据源: HuggingFace多粒度 (间隔: {args.interval})")
    elif args.use_kaggle:
        logger.info(f"📊 数据源: Kaggle历史数据")
    elif args.use_pipeline:
        sources = ["BTC价格"]
        if include_macro and APIConfig.FMP_API_KEY:
            sources.append("宏观经济")
        if include_onchain:
            sources.append("链上数据")
        sources.append("跨市场资产")
        logger.info(f"📊 数据源: 多源管道 ({', '.join(sources)})")
        logger.info(f"   数据天数: {args.fmp_days}天")
    elif args.use_fmp:
        logger.info(f"📊 数据源: FMP ({args.fmp_days}天)")
    elif args.use_hf:
        logger.info("📊 数据源: HuggingFace" + (" + CoinGecko最新数据" if args.merge_recent else ""))
    else:
        logger.info("📊 数据源: CoinGecko (90天)")
    
    # 获取数据
    try:
        df = asyncio.run(fetch_data(
            use_hf=args.use_hf, 
            merge_recent=args.merge_recent,
            use_fmp=args.use_fmp,
            fmp_days=args.fmp_days,
            use_pipeline=args.use_pipeline,
            include_macro=include_macro,
            include_onchain=include_onchain,
            use_hf_multi=args.use_hf_multi,
            interval=args.interval,
            use_binance_hist=args.use_binance_hist,
            use_kaggle=args.use_kaggle,
            days=args.days
        ))
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
        model.save(model_dir / 'gru_best.pth')  # 修正为 dashboard 期望的名称
        logger.info(f"✅ GRU 模型已保存到: {model_dir / 'gru_best.pth'}")
    
    if args.model in ['bilstm', 'all']:
        model, _ = train_bilstm(X_train, y_train, X_val, y_val)
        models['bilstm'] = model
        evaluate_model(model, X_test, y_test, 'BiLSTM')
        model.save(model_dir / 'bilstm_best.pth')
        logger.info(f"✅ BiLSTM 模型已保存到: {model_dir / 'bilstm_best.pth'}")
    
    if args.model in ['cnn_lstm', 'all']:
        model, _ = train_cnn_lstm(X_train, y_train, X_val, y_val)
        models['cnn_lstm'] = model
        evaluate_model(model, X_test, y_test, 'CNN-LSTM')
        model.save(model_dir / 'cnn_lstm_best.pth')
        logger.info(f"✅ CNN-LSTM 模型已保存到: {model_dir / 'cnn_lstm_best.pth'}")
    
    if args.model in ['lightgbm', 'all']:
        model, _ = train_lightgbm(X_train, y_train, X_val, y_val)
        models['lightgbm'] = model
        evaluate_model(model, X_test, y_test, 'LightGBM')
        model.save(model_dir / 'lightgbm_best.txt')  # 修正为 dashboard 期望的名称
        logger.info(f"✅ LightGBM 模型已保存到: {model_dir / 'lightgbm_best.txt'}")
    
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
