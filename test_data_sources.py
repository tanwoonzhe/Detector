"""
测试 HuggingFace 数据加载和 Binance 实时数据
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


async def test_hf_data():
    """测试 HuggingFace 数据加载"""
    print("\n" + "="*60)
    print("🧪 测试 1: HuggingFace 数据加载")
    print("="*60)
    
    try:
        from src.data_collection.hf_loader_fixed import load_hf_btc_data
        
        print("\n📥 尝试加载 HuggingFace 数据...")
        df = load_hf_btc_data()
        
        if not df.empty:
            print(f"✅ 成功加载 {len(df)} 条记录")
            print(f"   时间范围: {df.index.min()} ~ {df.index.max()}")
            print(f"\n   前5条数据:")
            print(df.head().to_string())
            print(f"\n   后5条数据:")
            print(df.tail().to_string())
            return True
        else:
            print("❌ 数据为空")
            return False
            
    except ImportError as e:
        print(f"⚠️  datasets 库未安装: {e}")
        print("   运行: pip install datasets")
        return False
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_binance_public():
    """测试 Binance 公开 API"""
    print("\n" + "="*60)
    print("🧪 测试 2: Binance 公开 API")
    print("="*60)
    
    try:
        from src.data_collection.binance_public import BinancePublicAPI
        
        api = BinancePublicAPI()
        
        # 测试实时价格
        print("\n📊 测试实时价格...")
        price_data = await api.get_current_price("BTCUSDT")
        print(f"✅ BTC 价格: ${price_data['price']:,.2f}")
        print(f"   时间: {price_data['timestamp']}")
        
        # 测试 24h 统计
        print("\n📈 测试 24小时统计...")
        ticker = await api.get_ticker_24h("BTCUSDT")
        print(f"✅ 当前价格: ${ticker['price']:,.2f}")
        print(f"   24h 涨跌: ${ticker['change']:+,.2f} ({ticker['change_percent']:+.2f}%)")
        print(f"   24h 最高: ${ticker['high']:,.2f}")
        print(f"   24h 最低: ${ticker['low']:,.2f}")
        
        # 测试 K 线
        print("\n📉 测试 K线数据...")
        df = await api.get_klines("BTCUSDT", "1h", days=1)
        print(f"✅ 获取 {len(df)} 条 1小时 K线")
        print(f"   时间范围: {df.index.min()} ~ {df.index.max()}")
        print(f"\n   最新 3 条:")
        print(df.tail(3).to_string())
        
        await api.close()
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_train_data_loading():
    """测试训练脚本的数据加载"""
    print("\n" + "="*60)
    print("🧪 测试 3: 训练脚本数据加载逻辑")
    print("="*60)
    
    try:
        # 模拟 train.py 的 fetch_data 函数
        from src.data_collection.coingecko_fetcher import CoinGeckoFetcher
        from src.data_collection.hf_loader_fixed import load_hf_btc_data
        import pandas as pd
        
        # 测试 CoinGecko 数据
        print("\n📊 测试 CoinGecko 数据...")
        fetcher = CoinGeckoFetcher()
        market_data = await fetcher.get_hourly_ohlcv("bitcoin", "usd", days=7)
        df_cg = market_data.to_dataframe()
        await fetcher.close()
        print(f"✅ CoinGecko: {len(df_cg)} 条")
        
        # 测试 HF 数据
        print("\n📥 测试 HuggingFace 数据...")
        try:
            df_hf = load_hf_btc_data()
            if not df_hf.empty:
                print(f"✅ HuggingFace: {len(df_hf)} 条")
                
                # 测试合并
                print("\n🔗 测试数据合并...")
                df_recent = df_cg[df_cg.index > df_hf.index.max()]
                if not df_recent.empty:
                    df_merged = pd.concat([df_hf, df_recent]).sort_index()
                    print(f"✅ 合并成功: {len(df_merged)} 条")
                    print(f"   HF: {len(df_hf)} + 最新: {len(df_recent)} = 总计: {len(df_merged)}")
                else:
                    print("⚠️  没有新数据需要合并")
            else:
                print("⚠️  HF 数据为空，将使用 CoinGecko")
        except Exception as e:
            print(f"⚠️  HF 数据加载失败: {e}")
            print("   将回退到 CoinGecko 数据")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """运行所有测试"""
    print("\n" + "="*60)
    print("🚀 数据源测试套件")
    print("="*60)
    
    results = []
    
    # 测试 1: HuggingFace
    hf_ok = await test_hf_data()
    results.append(("HuggingFace 数据", hf_ok))
    
    # 测试 2: Binance
    binance_ok = await test_binance_public()
    results.append(("Binance 实时数据", binance_ok))
    
    # 测试 3: 训练数据加载
    train_ok = await test_train_data_loading()
    results.append(("训练数据加载", train_ok))
    
    # 总结
    print("\n" + "="*60)
    print("📊 测试总结")
    print("="*60)
    
    for name, ok in results:
        status = "✅ 通过" if ok else "❌ 失败"
        print(f"{status} - {name}")
    
    all_pass = all(ok for _, ok in results)
    
    if all_pass:
        print("\n🎉 所有测试通过!")
    else:
        print("\n⚠️  部分测试失败，但不影响基本功能")
    
    print("\n💡 使用建议:")
    if results[0][1]:  # HF 测试通过
        print("   ✅ HuggingFace 可用 - 训练时使用 --use-hf 获取历史数据")
    else:
        print("   ⚠️  HuggingFace 不可用 - 使用 CoinGecko (90天数据)")
    
    if results[1][1]:  # Binance 测试通过
        print("   ✅ Binance 可用 - Dashboard 可显示实时价格")
    else:
        print("   ⚠️  Binance 不可用 - 使用 CoinGecko (小时级)")
    
    print("\n")


if __name__ == "__main__":
    asyncio.run(main())
