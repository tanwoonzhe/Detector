"""
测试 Dashboard 修复
"""
import asyncio
from src.data_collection.binance_public import BinancePublicAPI


async def test_binance_api():
    """测试 Binance API"""
    print("测试 Binance API...")
    api = BinancePublicAPI()
    
    try:
        # 测试获取价格
        print("\n1. 获取当前价格...")
        price = await api.get_current_price("BTCUSDT")
        print(f"   成功: {price}")
        
        # 测试获取24h数据
        print("\n2. 获取24h数据...")
        ticker = await api.get_ticker_24h("BTCUSDT")
        print(f"   成功: {ticker}")
        
        # 测试获取K线
        print("\n3. 获取K线数据...")
        klines = await api.get_klines("BTCUSDT", "1h", days=1)
        print(f"   成功: 获取了 {len(klines)} 条K线数据")
        
        print("\n✅ 所有测试通过！")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await api.close()


def test_hf_loader():
    """测试 HuggingFace 加载器"""
    print("\n" + "="*50)
    print("测试 HuggingFace 数据加载...")
    print("="*50)
    
    try:
        from src.data_collection.hf_loader_fixed import load_hf_btc_data
        
        # 测试加载（如果缓存存在就很快）
        df = load_hf_btc_data()
        
        if not df.empty:
            print(f"\n✅ 加载成功!")
            print(f"   数据行数: {len(df)}")
            print(f"   时间范围: {df.index.min()} ~ {df.index.max()}")
            print(f"\n   数据列: {df.columns.tolist()}")
            print(f"\n   前5行:")
            print(df.head())
        else:
            print("\n⚠️ 返回空数据框")
            
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("="*50)
    print("Dashboard 修复测试")
    print("="*50)
    
    # 测试 Binance API
    asyncio.run(test_binance_api())
    
    # 测试 HuggingFace
    test_hf_loader()
    
    print("\n" + "="*50)
    print("测试完成")
    print("="*50)
