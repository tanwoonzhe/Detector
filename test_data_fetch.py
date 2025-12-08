"""
测试训练数据获取 - 诊断版
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.data_collection.coingecko_fetcher import CoinGeckoFetcher


async def test_data_fetch():
    """测试数据获取"""
    print("=" * 60)
    print("测试 CoinGecko 数据获取")
    print("=" * 60)
    
    fetcher = CoinGeckoFetcher()
    
    # 测试不同天数
    for days in [7, 30, 90]:
        print(f"\n📊 获取 {days} 天数据...")
        try:
            market_data = await fetcher.get_hourly_ohlcv(
                symbol="bitcoin",
                days=days,
                vs_currency="usd"
            )
            
            df = market_data.to_dataframe()
            print(f"✅ 成功: {len(df)} 条记录")
            print(f"   时间范围: {df.index.min()} ~ {df.index.max()}")
            print(f"   价格范围: ${df['close'].min():.2f} ~ ${df['close'].max():.2f}")
            
            # 检查是否有足够的数据用于特征工程
            if len(df) >= 100:
                print(f"   ✅ 数据量充足（>= 100 行）")
            else:
                print(f"   ⚠️  数据量较少，特征工程可能失败")
        
        except Exception as e:
            print(f"❌ 失败: {e}")
    
    await fetcher.close()
    
    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)
    print("\n建议:")
    print("  • 如果数据量 < 100 行，特征工程会删除太多行导致为空")
    print("  • 使用至少 30 天数据进行训练")
    print("  • 或者使用 HuggingFace 数据集（更多历史数据）")
    print("\n训练命令:")
    print("  python train.py --model gru --epochs 100  # 使用 CoinGecko 90天数据")
    print("  python train.py --model gru --epochs 100 --use-hf  # 使用 HF 历史数据")


if __name__ == "__main__":
    asyncio.run(test_data_fetch())
