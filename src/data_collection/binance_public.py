"""
Binance 公开 API 数据获取器（无需 API Key）
================================
使用 Binance 公开市场数据 API，免费获取实时价格和 K 线数据

特点:
- ✅ 完全免费，无需 API Key
- ✅ 实时价格（秒级更新）
- ✅ K线数据（1分钟、5分钟、1小时等）
- ✅ 无速率限制（公开端点）
"""

import asyncio
import aiohttp
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)


class BinancePublicAPI:
    """
    Binance 公开 API 客户端
    
    使用方法:
        api = BinancePublicAPI()
        price = await api.get_current_price("BTCUSDT")
        klines = await api.get_klines("BTCUSDT", "1h", days=7)
        await api.close()
    """
    
    def __init__(self):
        self.base_url = "https://api.binance.com/api/v3"
        self._session: Optional[aiohttp.ClientSession] = None
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """获取或创建 HTTP 会话"""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session
    
    async def close(self):
        """关闭会话"""
        if self._session and not self._session.closed:
            await self._session.close()
    
    async def get_current_price(self, symbol: str = "BTCUSDT") -> dict:
        """
        获取当前实时价格（秒级更新）
        
        Args:
            symbol: 交易对，如 "BTCUSDT"
            
        Returns:
            {
                'symbol': 'BTCUSDT',
                'price': 43250.50,
                'timestamp': '2025-12-09 12:34:56'
            }
        """
        session = await self._get_session()
        url = f"{self.base_url}/ticker/price"
        
        try:
            async with session.get(url, params={"symbol": symbol}, timeout=aiohttp.ClientTimeout(total=10)) as response:
                if response.status == 451:
                    raise Exception("Binance API 访问受限（451错误）。可能是地区限制。")
                response.raise_for_status()
                data = await response.json()
                
                return {
                    'symbol': data['symbol'],
                    'price': float(data['price']),
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
        except Exception as e:
            logger.error(f"获取价格失败: {e}")
            raise
    
    async def get_ticker_24h(self, symbol: str = "BTCUSDT") -> dict:
        """
        获取 24 小时价格变动统计
        
        Returns:
            {
                'symbol': 'BTCUSDT',
                'price': 43250.50,
                'change': 1250.30,
                'change_percent': 2.98,
                'high': 43500.00,
                'low': 42000.00,
                'volume': 12345.67
            }
        """
        session = await self._get_session()
        url = f"{self.base_url}/ticker/24hr"
        
        try:
            async with session.get(url, params={"symbol": symbol}, timeout=aiohttp.ClientTimeout(total=10)) as response:
                if response.status == 451:
                    raise Exception("Binance API 访问受限（451错误）。可能是地区限制，请使用 VPN 或切换到 CoinGecko。")
                response.raise_for_status()
                data = await response.json()
                
                return {
                    'symbol': data['symbol'],
                    'price': float(data['lastPrice']),
                    'change': float(data['priceChange']),
                    'change_percent': float(data['priceChangePercent']),
                    'high': float(data['highPrice']),
                    'low': float(data['lowPrice']),
                    'volume': float(data['volume']),
                    'quote_volume': float(data['quoteVolume']),
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
        except Exception as e:
            logger.error(f"获取24h统计失败: {e}")
            raise
    
    async def get_klines(
        self, 
        symbol: str = "BTCUSDT",
        interval: str = "1h",
        days: int = 7,
        limit: int = 1000
    ) -> pd.DataFrame:
        """
        获取 K 线数据
        
        Args:
            symbol: 交易对
            interval: K线周期
                - 1m, 3m, 5m, 15m, 30m (分钟)
                - 1h, 2h, 4h, 6h, 8h, 12h (小时)
                - 1d, 3d, 1w, 1M (天/周/月)
            days: 获取多少天的历史数据
            limit: 最大返回条数（Binance 限制 1000）
            
        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        session = await self._get_session()
        url = f"{self.base_url}/klines"
        
        # 计算起始时间
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)
        
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": int(start_time.timestamp() * 1000),
            "endTime": int(end_time.timestamp() * 1000),
            "limit": limit
        }
        
        try:
            async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=30)) as response:
                if response.status == 451:
                    raise Exception("Binance API 访问受限（451错误）。可能是地区限制。")
                response.raise_for_status()
                klines = await response.json()
                
                # 转换为 DataFrame
                df = pd.DataFrame(klines, columns=[
                    'timestamp', 'open', 'high', 'low', 'close', 'volume',
                    'close_time', 'quote_volume', 'trades', 
                    'taker_buy_base', 'taker_buy_quote', 'ignore'
                ])
                
                # 只保留需要的列
                df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
                
                # 转换数据类型
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
                df = df.set_index('timestamp')
                
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    df[col] = df[col].astype(float)
                
                logger.info(f"✅ 获取 {symbol} {interval} K线: {len(df)} 条")
                
                return df
                
        except Exception as e:
            logger.error(f"获取K线失败: {e}")
            raise
    
    async def get_orderbook(self, symbol: str = "BTCUSDT", limit: int = 100) -> dict:
        """
        获取订单簿（深度数据）
        
        Args:
            symbol: 交易对
            limit: 深度档位 (5, 10, 20, 50, 100, 500, 1000, 5000)
            
        Returns:
            {
                'bids': [[price, quantity], ...],  # 买单
                'asks': [[price, quantity], ...]   # 卖单
            }
        """
        session = await self._get_session()
        url = f"{self.base_url}/depth"
        
        try:
            async with session.get(url, params={"symbol": symbol, "limit": limit}) as response:
                response.raise_for_status()
                data = await response.json()
                
                return {
                    'bids': [[float(p), float(q)] for p, q in data['bids']],
                    'asks': [[float(p), float(q)] for p, q in data['asks']],
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
        except Exception as e:
            logger.error(f"获取订单簿失败: {e}")
            raise


# 便捷函数
async def get_btc_realtime_price() -> float:
    """获取 BTC 实时价格（便捷函数）"""
    api = BinancePublicAPI()
    try:
        data = await api.get_current_price("BTCUSDT")
        return data['price']
    finally:
        await api.close()


async def get_btc_klines(interval: str = "1h", days: int = 7) -> pd.DataFrame:
    """获取 BTC K线数据（便捷函数）"""
    api = BinancePublicAPI()
    try:
        return await api.get_klines("BTCUSDT", interval, days)
    finally:
        await api.close()


# 测试代码
if __name__ == "__main__":
    async def test():
        api = BinancePublicAPI()
        
        print("\n" + "="*60)
        print("🧪 测试 Binance 公开 API")
        print("="*60)
        
        # 测试 1: 获取实时价格
        print("\n📊 测试 1: 实时价格")
        price_data = await api.get_current_price("BTCUSDT")
        print(f"   BTC 价格: ${price_data['price']:,.2f}")
        print(f"   时间: {price_data['timestamp']}")
        
        # 测试 2: 获取 24h 统计
        print("\n📈 测试 2: 24小时统计")
        ticker = await api.get_ticker_24h("BTCUSDT")
        print(f"   当前价格: ${ticker['price']:,.2f}")
        print(f"   24h 涨跌: ${ticker['change']:+,.2f} ({ticker['change_percent']:+.2f}%)")
        print(f"   24h 最高: ${ticker['high']:,.2f}")
        print(f"   24h 最低: ${ticker['low']:,.2f}")
        print(f"   24h 成交量: {ticker['volume']:,.2f} BTC")
        
        # 测试 3: 获取 K 线
        print("\n📉 测试 3: K线数据")
        df = await api.get_klines("BTCUSDT", "1h", days=1)
        print(f"   数据条数: {len(df)}")
        print(f"   时间范围: {df.index.min()} ~ {df.index.max()}")
        print(f"\n   最新 5 条:")
        print(df.tail().to_string())
        
        await api.close()
        
        print("\n" + "="*60)
        print("✅ 所有测试通过！")
        print("="*60)
    
    asyncio.run(test())
