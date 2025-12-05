"""
CoinGecko数据获取器
================================
实现CoinGecko API的数据获取，包含速率限制和缓存支持
"""

import asyncio
import aiohttp
import logging
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
import time

from .base import (
    DataFetcher, 
    DataSource, 
    OHLCV, 
    MarketData, 
    RateLimitInfo,
    DataFetcherFactory
)
from .cache import cache_manager
from config import APIConfig, TradingConfig

logger = logging.getLogger(__name__)


class RateLimiter:
    """速率限制器"""
    
    def __init__(self, calls_per_minute: int = 25):
        self.calls_per_minute = calls_per_minute
        self.min_interval = 60.0 / calls_per_minute
        self.last_call_time = 0.0
        self._lock = asyncio.Lock()
    
    async def acquire(self):
        """获取请求许可"""
        async with self._lock:
            current_time = time.time()
            elapsed = current_time - self.last_call_time
            wait_time = self.min_interval - elapsed
            
            if wait_time > 0:
                logger.debug(f"速率限制: 等待 {wait_time:.2f} 秒")
                await asyncio.sleep(wait_time)
            
            self.last_call_time = time.time()


class CoinGeckoFetcher(DataFetcher):
    """
    CoinGecko数据获取器
    
    特点:
    - 免费版支持30次/分钟请求
    - 2-90天历史数据返回小时粒度
    - 超过90天返回日粒度
    """
    
    def __init__(self):
        super().__init__(DataSource.COINGECKO)
        self.base_url = APIConfig.COINGECKO_BASE_URL
        self.rate_limiter = RateLimiter(APIConfig.COINGECKO_RATE_LIMIT)
        self._session: Optional[aiohttp.ClientSession] = None
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """获取或创建HTTP会话"""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                headers={"Accept": "application/json"}
            )
        return self._session
    
    async def _request(self, endpoint: str, params: Optional[Dict] = None) -> Dict:
        """
        发送API请求
        
        Args:
            endpoint: API端点
            params: 请求参数
            
        Returns:
            JSON响应
        """
        await self.rate_limiter.acquire()
        
        session = await self._get_session()
        url = f"{self.base_url}{endpoint}"
        
        try:
            async with session.get(url, params=params) as response:
                cache_manager.record_api_request(
                    self.source.value, 
                    endpoint, 
                    response.status == 200
                )
                
                if response.status == 429:
                    logger.warning("CoinGecko API速率限制，等待60秒...")
                    await asyncio.sleep(60)
                    return await self._request(endpoint, params)
                
                response.raise_for_status()
                return await response.json()
                
        except aiohttp.ClientError as e:
            logger.error(f"CoinGecko API请求失败: {e}")
            raise
    
    async def get_hourly_ohlcv(
        self, 
        symbol: str = "bitcoin",
        days: int = 90,
        vs_currency: str = "usd"
    ) -> MarketData:
        """
        获取小时级OHLCV数据
        
        CoinGecko market_chart端点:
        - days=1: 5分钟粒度
        - days=2-90: 小时粒度
        - days>90: 日粒度
        
        Args:
            symbol: CoinGecko币种ID (如 "bitcoin")
            days: 历史天数 (建议2-90以获取小时数据)
            vs_currency: 计价货币
            
        Returns:
            MarketData对象
        """
        # 先检查缓存
        cached_df = cache_manager.get_ohlcv(
            symbol, 
            self.source.value,
            start_time=datetime.now() - timedelta(days=days)
        )
        
        # 如果缓存数据足够新（最近1小时内有更新），直接返回
        if not cached_df.empty:
            latest_cached = cached_df.index.max()
            if datetime.now() - latest_cached.to_pydatetime() < timedelta(hours=1):
                logger.info(f"使用缓存数据: {len(cached_df)} 条记录")
                return self._df_to_market_data(symbol, cached_df)
        
        # 获取新数据
        logger.info(f"从CoinGecko获取 {symbol} 的 {days} 天历史数据...")
        
        data = await self._request(
            f"/coins/{symbol}/market_chart",
            params={
                "vs_currency": vs_currency,
                "days": str(days),
                "interval": "hourly" if days <= 90 else "daily"
            }
        )
        
        # 解析数据
        ohlcv_list = self._parse_market_chart(data)
        
        # 获取额外市场信息
        coin_data = await self._request(f"/coins/{symbol}")
        market_data = MarketData(
            symbol=symbol,
            ohlcv_data=ohlcv_list,
            market_cap=coin_data.get("market_data", {}).get("market_cap", {}).get(vs_currency),
            total_volume_24h=coin_data.get("market_data", {}).get("total_volume", {}).get(vs_currency),
            price_change_24h=coin_data.get("market_data", {}).get("price_change_percentage_24h"),
            metadata={
                "source": self.source.value,
                "fetched_at": datetime.now().isoformat(),
                "days": days
            }
        )
        
        # 保存到缓存
        df = market_data.to_dataframe()
        if not df.empty:
            cache_manager.save_ohlcv(symbol, self.source.value, df)
        
        logger.info(f"获取到 {len(ohlcv_list)} 条OHLCV数据")
        return market_data
    
    def _parse_market_chart(self, data: Dict) -> List[OHLCV]:
        """
        解析market_chart响应
        
        CoinGecko返回格式:
        {
            "prices": [[timestamp, price], ...],
            "market_caps": [[timestamp, cap], ...],
            "total_volumes": [[timestamp, volume], ...]
        }
        
        注意: CoinGecko不直接提供OHLC，需要用价格数据估算
        """
        prices = data.get("prices", [])
        volumes = data.get("total_volumes", [])
        
        if not prices:
            return []
        
        ohlcv_list = []
        
        # 创建时间戳到成交量的映射
        volume_map = {int(v[0]): v[1] for v in volumes}
        
        # 按小时分组价格数据
        hourly_data: Dict[datetime, List[float]] = {}
        for timestamp_ms, price in prices:
            dt = datetime.fromtimestamp(timestamp_ms / 1000)
            # 对齐到小时
            hour_dt = dt.replace(minute=0, second=0, microsecond=0)
            
            if hour_dt not in hourly_data:
                hourly_data[hour_dt] = []
            hourly_data[hour_dt].append(price)
        
        # 生成OHLCV
        for hour_dt, hour_prices in sorted(hourly_data.items()):
            if not hour_prices:
                continue
            
            # 估算OHLC
            open_price = hour_prices[0]
            close_price = hour_prices[-1]
            high_price = max(hour_prices)
            low_price = min(hour_prices)
            
            # 获取对应成交量
            timestamp_ms = int(hour_dt.timestamp() * 1000)
            volume = volume_map.get(timestamp_ms, 0)
            
            ohlcv_list.append(OHLCV(
                timestamp=hour_dt,
                open=open_price,
                high=high_price,
                low=low_price,
                close=close_price,
                volume=volume
            ))
        
        return ohlcv_list
    
    def _df_to_market_data(self, symbol: str, df) -> MarketData:
        """DataFrame转MarketData"""
        ohlcv_list = []
        for timestamp, row in df.iterrows():
            ohlcv_list.append(OHLCV(
                timestamp=timestamp.to_pydatetime() if hasattr(timestamp, 'to_pydatetime') else timestamp,
                open=row['open'],
                high=row['high'],
                low=row['low'],
                close=row['close'],
                volume=row['volume']
            ))
        
        return MarketData(
            symbol=symbol,
            ohlcv_data=ohlcv_list,
            metadata={"source": "cache"}
        )
    
    async def get_current_price(self, symbol: str = "bitcoin") -> float:
        """获取当前价格"""
        data = await self._request(
            "/simple/price",
            params={
                "ids": symbol,
                "vs_currencies": "usd"
            }
        )
        return data.get(symbol, {}).get("usd", 0.0)
    
    async def get_ohlc(
        self, 
        symbol: str = "bitcoin",
        days: int = 7,
        vs_currency: str = "usd"
    ) -> List[OHLCV]:
        """
        获取OHLC蜡烛图数据（官方OHLC端点）
        
        注意: 此端点返回的粒度固定:
        - days=1: 30分钟
        - days=7: 4小时
        - days=14: 4小时
        - days=30: 日
        - days=90/180/365/max: 3天
        """
        data = await self._request(
            f"/coins/{symbol}/ohlc",
            params={
                "vs_currency": vs_currency,
                "days": str(days)
            }
        )
        
        ohlcv_list = []
        for item in data:
            timestamp_ms, open_p, high_p, low_p, close_p = item
            ohlcv_list.append(OHLCV(
                timestamp=datetime.fromtimestamp(timestamp_ms / 1000),
                open=open_p,
                high=high_p,
                low=low_p,
                close=close_p,
                volume=0  # OHLC端点不返回成交量
            ))
        
        return ohlcv_list
    
    def get_rate_limit(self) -> RateLimitInfo:
        """获取速率限制信息"""
        recent_calls = cache_manager.get_recent_request_count(self.source.value, minutes=1)
        return RateLimitInfo(
            calls_per_minute=APIConfig.COINGECKO_RATE_LIMIT,
            remaining_calls=max(0, APIConfig.COINGECKO_RATE_LIMIT - recent_calls)
        )
    
    async def health_check(self) -> bool:
        """检查API连接状态"""
        try:
            await self._request("/ping")
            return True
        except Exception as e:
            logger.error(f"CoinGecko健康检查失败: {e}")
            return False
    
    async def close(self):
        """关闭HTTP会话"""
        if self._session and not self._session.closed:
            await self._session.close()


# 注册到工厂
DataFetcherFactory.register(DataSource.COINGECKO, CoinGeckoFetcher)
