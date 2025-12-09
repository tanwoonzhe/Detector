"""
快速训练测试脚本 - 诊断特征工程问题
"""
import asyncio
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from config import FeatureConfig
from src.data_collection.coingecko_fetcher import CoinGeckoFetcher
from src.features import FeatureEngineer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_training_pipeline():
    """测试训练流程"""
    print("=" * 70)
    print("测试训练数据流程")
    print("=" * 70)
    
    # 步骤1: 获取数据
    print("\n步骤1: 获取 CoinGecko 数据...")
    fetcher = CoinGeckoFetcher()
    market_data = await fetcher.get_hourly_ohlcv(
        symbol="bitcoin",
        vs_currency="usd",
        days=90
    )
    df = market_data.to_dataframe()
    await fetcher.close()
    
    print(f"✅ 原始数据: {len(df)} 行")
    print(f"   时间范围: {df.index.min()} ~ {df.index.max()}")
    print(f"   列: {list(df.columns)}")
    
    # 检查数据完整性
    print(f"\n数据完整性检查:")
    print(f"  缺失值: {df.isna().sum().sum()}")
    print(f"  重复行: {df.duplicated().sum()}")
    
    # 步骤2: 特征工程
    print(f"\n步骤2: 特征工程...")
    print(f"  当前配置:")
    print(f"    SMA_PERIODS: {FeatureConfig.SMA_PERIODS}")
    print(f"    EMA_PERIODS: {FeatureConfig.EMA_PERIODS}")
    print(f"    RETURN_PERIODS: {FeatureConfig.RETURN_PERIODS}")
    if hasattr(FeatureConfig, 'TECHNICAL_WINDOWS'):
        print(f"    TECHNICAL_WINDOWS: {FeatureConfig.TECHNICAL_WINDOWS}")
    
    engineer = FeatureEngineer()
    
    try:
        df_features = engineer.create_features(df)
        print(f"\n✅ 特征工程成功")
        print(f"   输出数据: {len(df_features)} 行 ({len(df) - len(df_features)} 行被删除)")
        print(f"   特征数量: {len(df_features.columns)} 列")
        print(f"   删除比例: {(len(df) - len(df_features)) / len(df) * 100:.1f}%")
        
        if len(df_features) >= 100:
            print(f"\n✅ 数据充足，可以开始训练！")
            print(f"   建议使用命令:")
            print(f"   python train.py --model gru --epochs 50")
        else:
            print(f"\n❌ 数据不足 ({len(df_features)} < 100 行)")
            print(f"   无法训练模型")
        
        # 显示前几列
        print(f"\n前5行数据:")
        print(df_features.head())
        
        # 检查缺失值
        nan_cols = df_features.isna().sum()
        if nan_cols.sum() > 0:
            print(f"\n⚠️  仍有缺失值的列:")
            print(nan_cols[nan_cols > 0].head(10))
    
    except Exception as e:
        print(f"\n❌ 特征工程失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 步骤3: 创建标签
    print(f"\n步骤3: 创建训练标签...")
    try:
        df_labeled = engineer.create_labels(df_features)
        print(f"✅ 标签创建成功: {len(df_labeled)} 行")
        
        # 检查标签分布
        for window in [0.5, 1, 2, 4]:
            window_label = str(int(window)) if float(window).is_integer() else str(window).rstrip('0').rstrip('.')
            target_col = f'target_{window_label}h'
            if target_col in df_labeled.columns:
                value_counts = df_labeled[target_col].value_counts()
                print(f"  {target_col}: {value_counts.to_dict()}")
    
    except Exception as e:
        print(f"❌ 标签创建失败: {e}")
        return False
    
    print("\n" + "=" * 70)
    print("测试完成！")
    print("=" * 70)
    return True


if __name__ == "__main__":
    success = asyncio.run(test_training_pipeline())
    sys.exit(0 if success else 1)
