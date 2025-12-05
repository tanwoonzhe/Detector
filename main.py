"""
BTC趋势预测系统 - 主入口
================================
提供命令行接口运行预测和Dashboard

使用方法:
    # 启动Dashboard
    python main.py --dashboard
    
    # 单次预测
    python main.py --predict
    
    # 训练模型
    python main.py --train --model gru
"""

import argparse
import asyncio
import logging
import sys
import os
from pathlib import Path
from datetime import datetime

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from config import ModelConfig, TradingConfig

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('btc_predictor.log')
    ]
)
logger = logging.getLogger(__name__)


def run_dashboard():
    """启动Streamlit Dashboard"""
    import subprocess
    
    dashboard_path = Path(__file__).parent / 'app' / 'dashboard.py'
    
    logger.info("启动Dashboard...")
    logger.info("访问地址: http://localhost:8501")
    
    subprocess.run([
        sys.executable, '-m', 'streamlit', 'run',
        str(dashboard_path),
        '--server.port', '8501',
        '--server.headless', 'true'
    ])


async def run_prediction():
    """运行单次预测"""
    from src.data_collection.coingecko_fetcher import CoinGeckoFetcher
    from src.features import FeatureEngineer
    from src.models import GRUPredictor
    from src.signals import SignalGenerator, SignalFormatter
    
    logger.info("="*50)
    logger.info("BTC趋势预测")
    logger.info("="*50)
    
    # 获取最新数据
    logger.info("获取最新数据...")
    fetcher = CoinGeckoFetcher()
    
    try:
        market_data = await fetcher.get_hourly_ohlcv("bitcoin", days=7)
    except Exception as e:
        logger.error(f"获取数据失败: {e}")
        return
    
    if not market_data.ohlcv_data:
        logger.error("没有获取到数据")
        return
    
    # 转换为DataFrame
    import pandas as pd
    df = market_data.to_dataframe()
    
    # 显示当前价格
    current_price = df['close'].iloc[-1]
    price_change = (df['close'].iloc[-1] / df['close'].iloc[-2] - 1) * 100
    
    print(f"\n当前价格: ${current_price:,.2f}")
    print(f"24h变化: {price_change:+.2f}%")
    
    # 特征工程
    logger.info("处理特征...")
    engineer = FeatureEngineer()
    df_features = engineer.create_features(df)
    
    # 加载模型
    model_path = Path(__file__).parent / 'models' / 'saved' / 'gru_model.pt'
    
    if model_path.exists():
        logger.info("加载预训练模型...")
        model = GRUPredictor()
        model.build(
            input_shape=(ModelConfig.SEQUENCE_LENGTH, len(engineer.get_feature_columns(df_features))),
            n_classes=3
        )
        model.load(model_path)
        
        # 获取最新序列
        X = engineer.get_latest_sequence(df_features)
        
        # 预测
        proba = model.predict_proba(X)
        pred = model.predict(X)[0]
        
        # 生成信号
        signal_gen = SignalGenerator()
        
        # 各窗口预测 (这里简化为相同预测)
        predictions = {w: pred for w in TradingConfig.PREDICTION_WINDOWS}
        probabilities = {w: proba for w in TradingConfig.PREDICTION_WINDOWS}
        
        signal = signal_gen.generate_signal(
            predictions, 
            probabilities,
            sentiment_score=0,
            timestamp=datetime.now()
        )
        
        # 显示信号
        print(SignalFormatter.format_display(signal))
        
    else:
        logger.warning("模型文件不存在，请先训练模型")
        logger.info("运行: python train.py --model gru")
        
        # 显示简单分析
        print("\n技术分析:")
        if 'rsi' in df_features.columns:
            rsi = df_features['rsi'].iloc[-1]
            print(f"  RSI: {rsi:.1f}", end="")
            if rsi > 70:
                print(" (超买)")
            elif rsi < 30:
                print(" (超卖)")
            else:
                print(" (中性)")


def run_train(model: str, epochs: int):
    """训练模型"""
    import subprocess
    
    train_script = Path(__file__).parent / 'train.py'
    
    cmd = [
        sys.executable, str(train_script),
        '--model', model,
        '--epochs', str(epochs)
    ]
    
    subprocess.run(cmd)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='BTC趋势预测系统',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  启动Dashboard:      python main.py --dashboard
  单次预测:           python main.py --predict
  训练GRU模型:        python main.py --train --model gru
  训练所有模型:       python main.py --train --model all --epochs 50
        """
    )
    
    parser.add_argument('--dashboard', action='store_true',
                       help='启动Streamlit Dashboard')
    parser.add_argument('--predict', action='store_true',
                       help='运行单次预测')
    parser.add_argument('--train', action='store_true',
                       help='训练模型')
    parser.add_argument('--model', type=str, default='gru',
                       choices=['gru', 'bilstm', 'cnn_lstm', 'lightgbm', 'all'],
                       help='要训练的模型类型')
    parser.add_argument('--epochs', type=int, default=100,
                       help='训练轮数')
    
    args = parser.parse_args()
    
    # 打印欢迎信息
    print("""
╔══════════════════════════════════════════════════╗
║        🚀 BTC趋势预测系统 v1.0                    ║
║                                                  ║
║   基于深度学习的加密货币趋势预测与交易信号生成    ║
╚══════════════════════════════════════════════════╝
    """)
    
    if args.dashboard:
        run_dashboard()
    elif args.predict:
        asyncio.run(run_prediction())
    elif args.train:
        run_train(args.model, args.epochs)
    else:
        # 默认显示帮助
        parser.print_help()
        print("\n快速开始:")
        print("  1. 训练模型: python main.py --train --model gru")
        print("  2. 启动界面: python main.py --dashboard")
        print("  3. 单次预测: python main.py --predict")


if __name__ == "__main__":
    main()
