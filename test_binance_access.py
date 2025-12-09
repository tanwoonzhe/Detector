"""
测试 Binance API 访问
检查是否受到地区限制（451错误）
"""

import asyncio
import aiohttp
import sys


async def test_binance_access():
    """测试 Binance API 是否可访问"""
    
    print("\n" + "="*60)
    print("🧪 测试 Binance API 访问")
    print("="*60)
    
    base_url = "https://api.binance.com/api/v3"
    
    async with aiohttp.ClientSession() as session:
        # 测试 1: Ping
        print("\n📡 测试 1: API Ping")
        try:
            async with session.get(f"{base_url}/ping", timeout=aiohttp.ClientTimeout(total=5)) as response:
                if response.status == 200:
                    print("✅ Ping 成功 - Binance API 可访问")
                else:
                    print(f"⚠️  Ping 返回状态码: {response.status}")
        except asyncio.TimeoutError:
            print("❌ 超时 - 无法连接到 Binance")
            return False
        except Exception as e:
            print(f"❌ 错误: {e}")
            return False
        
        # 测试 2: 获取价格
        print("\n💰 测试 2: 获取 BTC 价格")
        try:
            async with session.get(
                f"{base_url}/ticker/price",
                params={"symbol": "BTCUSDT"},
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                if response.status == 451:
                    print("❌ 451 错误 - Binance 在你的地区受限")
                    print("\n可能原因:")
                    print("  1. 地区限制（某些国家/地区无法访问）")
                    print("  2. 网络防火墙")
                    print("  3. ISP 限制")
                    print("\n解决方案:")
                    print("  ✅ 使用 VPN 连接到允许地区")
                    print("  ✅ 使用 CoinGecko Dashboard 代替")
                    print("     命令: streamlit run app/dashboard_stable.py")
                    return False
                elif response.status == 200:
                    data = await response.json()
                    price = float(data['price'])
                    print(f"✅ 成功获取价格: ${price:,.2f}")
                else:
                    print(f"⚠️  返回状态码: {response.status}")
                    text = await response.text()
                    print(f"   响应: {text[:200]}")
        except asyncio.TimeoutError:
            print("❌ 超时 - 请求过慢")
            return False
        except Exception as e:
            print(f"❌ 错误: {e}")
            return False
        
        # 测试 3: 获取 K 线
        print("\n📊 测试 3: 获取 K 线数据")
        try:
            async with session.get(
                f"{base_url}/klines",
                params={
                    "symbol": "BTCUSDT",
                    "interval": "1h",
                    "limit": 5
                },
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                if response.status == 200:
                    klines = await response.json()
                    print(f"✅ 成功获取 {len(klines)} 条 K 线数据")
                elif response.status == 451:
                    print("❌ 451 错误 - K 线接口也受限")
                    return False
                else:
                    print(f"⚠️  返回状态码: {response.status}")
        except Exception as e:
            print(f"❌ 错误: {e}")
            return False
    
    print("\n" + "="*60)
    print("🎉 所有测试通过！Binance API 完全可用")
    print("="*60)
    print("\n你可以使用 Binance 实时 Dashboard:")
    print("  streamlit run app/dashboard_realtime_binance.py")
    print()
    
    return True


async def test_alternative_endpoints():
    """测试备用端点"""
    print("\n" + "="*60)
    print("🔄 测试 Binance 备用端点")
    print("="*60)
    
    endpoints = [
        "https://api.binance.com",
        "https://api1.binance.com",
        "https://api2.binance.com",
        "https://api3.binance.com",
    ]
    
    async with aiohttp.ClientSession() as session:
        for endpoint in endpoints:
            try:
                print(f"\n尝试: {endpoint}")
                async with session.get(
                    f"{endpoint}/api/v3/ping",
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    if response.status == 200:
                        print(f"  ✅ 可用")
                        return endpoint
                    else:
                        print(f"  ❌ 状态码: {response.status}")
            except Exception as e:
                print(f"  ❌ 失败: {str(e)[:50]}")
    
    return None


async def main():
    print("\n" + "="*60)
    print("🌐 Binance API 访问测试工具")
    print("="*60)
    
    # 主测试
    success = await test_binance_access()
    
    if not success:
        # 尝试备用端点
        print("\n正在尝试备用端点...")
        endpoint = await test_alternative_endpoints()
        
        if endpoint:
            print(f"\n✅ 找到可用端点: {endpoint}")
        else:
            print("\n" + "="*60)
            print("❌ 无法访问 Binance API")
            print("="*60)
            print("\n推荐方案:")
            print("\n1️⃣  使用 CoinGecko Dashboard (稳定可靠)")
            print("   streamlit run app/dashboard_stable.py")
            print("\n2️⃣  使用 VPN 后再试")
            print("   python test_binance_access.py")
            print("\n3️⃣  继续训练模型（不影响训练）")
            print("   python train.py --model gru --epochs 100")
            print()
            sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
