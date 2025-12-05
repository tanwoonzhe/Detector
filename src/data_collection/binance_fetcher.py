"""
Binance数据获取器（预留接口）
================================
为未来扩展Binance API支持预留的接口
"""

import asyncio
import aiohttp
import logging
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any

from .base import (
    DataFetcher, 
    DataSource, 
    OHLCV, 
    MarketData, 
    RateLimitInfo,
    DataFetcherFactory
)
from .cache import cache_manager
from config import APIConfig

logger = logging.getLogger(__name__)


class BinanceFetcher(DataFetcher):
    """
    Binance数据获取器（预留实现）
    
    特点:
    - 支持更细粒度数据（1分钟级别）
    - 更长历史数据
    - WebSocket实时数据支持
    
    注意: 此类为预留接口，完整实现需要API密钥
    """
    
    def __init__(self, api_key: str = "", secret_key: str = ""):
        super().__init__(DataSource.BINANCE)
        self.base_url = APIConfig.BINANCE_BASE_URL
        self.api_key = api_key or APIConfig.BINANCE_API_KEY
        self.secret_key = secret_key or APIConfig.BINANCE_SECRET_KEY
        self._session: Optional[aiohttp.ClientSession] = None
        
        logger.warning("Binance数据获取器为预留接口，需要API密钥才能使用")
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """获取或创建HTTP会话"""
        if self._session is None or self._session.closed:
            headers = {"Accept": "application/json"}
            if self.api_key:
                headers["X-MBX-APIKEY"] = self.api_key
            self._session = aiohttp.ClientSession(headers=headers)
        return self._session
    
    async def _request(self, endpoint: str, params: Optional[Dict] = None) -> Any:
        """发送API请求"""
        session = await self._get_session()
        url = f"{self.base_url}{endpoint}"
        
        try:
            async with session.get(url, params=params) as response:
                response.raise_for_status()
                return await response.json()
        except aiohttp.ClientError as e:
            logger.error(f"Binance API请求失败: {e}")
            raise
    
    async def get_hourly_ohlcv(
        self, 
        symbol: str = "BTCUSDT",
        days: int = 90,
        vs_currency: str = "usd"
    ) -> MarketData:
        """
        获取小时级OHLCV数据
        
        Binance klines端点:
        - 支持1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M
        - 每次最多1000条数据
        
        Args:
            symbol: Binance交易对 (如 "BTCUSDT")
            days: 历史天数
            vs_currency: 忽略（Binance使用交易对确定）
            
        Returns:
            MarketData对象
        """
        if not self.api_key:
            logger.warning("未配置Binance API密钥，返回空数据")
            return MarketData(symbol=symbol, ohlcv_data=[])
        
        # 计算时间范围
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)
        
        # 获取K线数据
        all_klines = []
        current_start = int(start_time.timestamp() * 1000)
        end_ms = int(end_time.timestamp() * 1000)
        
        while current_start < end_ms:
            data = await self._request(
                "/klines",
                params={
                    "symbol": symbol,
                    "interval": "1h",
                    "startTime": current_start,
                    "endTime": end_ms,
                    "limit": 1000
                }
            )
            
            if not data:
                break
            
            all_klines.extend(data)
            # 移动到下一批
            current_start = data[-1][0] + 1
        
        # 解析数据
        ohlcv_list = []
        for kline in all_klines:
            """
            Binance kline格式:
            [
                Open time,      # 0
                Open,           # 1
                High,           # 2
                Low,            # 3
                Close,          # 4
                Volume,         # 5
                Close time,     # 6
                Quote volume,   # 7
                Trades,         # 8
                Taker buy base, # 9
                Taker buy quote,# 10
                Ignore          # 11
            ]
            """
            ohlcv_list.append(OHLCV(
                timestamp=datetime.fromtimestamp(kline[0] / 1000),
                open=float(kline[1]),
                high=float(kline[2]),
                low=float(kline[3]),
                close=float(kline[4]),
                volume=float(kline[5])
            ))
        
        market_data = MarketData(
            symbol=symbol,
            ohlcv_data=ohlcv_list,
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
        
        return market_data
    
    async def get_current_price(self, symbol: str = "BTCUSDT") -> float:
        """获取当前价格"""
        if not self.api_key:
            return 0.0
        
        data = await self._request(
            "/ticker/price",
            params={"symbol": symbol}
        )
        return float(data.get("price", 0))
    
    def get_rate_limit(self) -> RateLimitInfo:
        """
        获取速率限制信息
        
        Binance使用权重系统:
        - 每分钟1200权重限制
        - 不同端点权重不同
        """
        return RateLimitInfo(
            calls_per_minute=1200,  # 权重限制
            daily_limit=None
        )
    
    async def health_check(self) -> bool:
        """检查API连接状态"""
        try:
            await self._request("/ping")
            return True
        except Exception as e:
            logger.error(f"Binance健康检查失败: {e}")
            return False
    
    async def close(self):
        """关闭HTTP会话"""
        if self._session and not self._session.closed:
            await self._session.close()


# 注册到工厂
DataFetcherFactory.register(DataSource.BINANCE, BinanceFetcher)
