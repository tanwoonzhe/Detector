"""
快速测试数据加载和特征工程修复
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.data_collection.coingecko_fetcher import CoinGeckoFetcher
from src.features.engineer import FeatureEngineer
from config import TradingConfig
import asyncio

async def test_data_and_features():
    print("=" * 50)
    print("测试 1: 数据获取")
    print("=" * 50)
    
    config = TradingConfig()
    fetcher = CoinGeckoFetcher()
    
    print("从 CoinGecko 获取数据...")
    ohlc_list = await fetcher.get_ohlc("bitcoin", days=90)
    
    # 转换为 DataFrame
    import pandas as pd
    df = pd.DataFrame([{
        'timestamp': ohlc.timestamp,
        'open': ohlc.open,
        'high': ohlc.high,
        'low': ohlc.low,
        'close': ohlc.close,
        'volume': ohlc.volume
    } for ohlc in ohlc_list])
    df = df.set_index('timestamp')
    
    print(f"✅ 成功获取 {len(df)} 行数据")
    print(f"时间范围: {df.index[0]} 到 {df.index[-1]}")
    print()
    
    print("=" * 50)
    print("测试 2: 特征工程")
    print("=" * 50)
    
    engineer = FeatureEngineer()
    print("生成特征...")
    df_features = engineer.create_features(df)
    
    print(f"✅ 特征工程完成!")
    print(f"初始数据: {len(df)} 行")
    print(f"处理后数据: {len(df_features)} 行")
    print(f"特征数量: {len(df_features.columns)}")
    print(f"数据保留率: {len(df_features)/len(df)*100:.1f}%")
    
    # 检查是否有NaN或Inf
    nan_count = df_features.isna().sum().sum()
    inf_count = np.isinf(df_features.select_dtypes(include=[np.number])).sum().sum()
    
    print()
    print("数据质量检查:")
    print(f"  NaN 值: {nan_count}")
    print(f"  Inf 值: {inf_count}")
    
    if len(df_features) > 0:
        print("\n✅ 测试通过: 数据处理正常!")
    else:
        print("\n❌ 测试失败: 所有数据被删除!")
    
    return df, df_features

if __name__ == "__main__":
    import numpy as np
    df, df_features = asyncio.run(test_data_and_features())
