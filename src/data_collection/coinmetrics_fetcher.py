"""
Coin Metrics 数据获取器
================================
获取链上数据和网络指标

Coin Metrics 提供:
- 链上指标 (hashrate, 活跃地址, 交易数, 手续费等)
- 网络健康状况
- 供应量数据
- 流通量统计

免费 API: https://docs.coinmetrics.io/api/v4
"""

import asyncio
import aiohttp
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
import pandas as pd

from .base import DataFetcher, DataSource, RateLimitInfo
from .cache import cache_manager

logger = logging.getLogger(__name__)


class CoinMetricsFetcher(DataFetcher):
    """
    Coin Metrics 链上数据获取器
    
    免费 API 特点:
    - 支持多种链上指标
    - 日级数据粒度
    - 无需 API 密钥（社区版）
    """
    
    # 支持的链上指标
    SUPPORTED_METRICS = [
        # 网络活动
        "AdrActCnt",        # 活跃地址数
        "TxCnt",            # 交易数量
        "TxTfrValAdjUSD",   # 调整后转账价值(USD)
        "TxTfrValMeanUSD",  # 平均转账价值(USD)
        
        # 挖矿/网络安全
        "HashRate",         # 哈希率
        "DiffMean",         # 平均难度
        "BlkCnt",           # 区块数
        "BlkSizeMeanByte",  # 平均区块大小
        
        # 供应量
        "SplyCur",          # 当前供应量
        "SplyAct1d",        # 1天活跃供应
        "SplyAct30d",       # 30天活跃供应
        
        # 费用
        "FeeMeanUSD",       # 平均手续费(USD)
        "FeeTotUSD",        # 总手续费(USD)
        
        # 市场
        "CapMrktCurUSD",    # 市值(USD)
        "CapRealUSD",       # 已实现市值(USD)
        "NVTAdj",           # NVT比率(调整后)
        "VelCur1yr",        # 流通速度
    ]
    
    def __init__(self, api_key: str = ""):
        """
        初始化 CoinMetrics Fetcher
        
        Args:
            api_key: API密钥（社区版可为空）
        """
        super().__init__(DataSource.COINGECKO)  # 临时使用，因为枚举中没有COINMETRICS
        self.api_key = api_key
        self.base_url = "https://community-api.coinmetrics.io/v4"
        self._session: Optional[aiohttp.ClientSession] = None
        self._last_request_time = 0.0
        self._min_interval = 1.0  # 每秒最多1个请求
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """获取HTTP会话"""
        if self._session is None or self._session.closed:
            headers = {"Accept": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            self._session = aiohttp.ClientSession(headers=headers)
        return self._session
    
    async def _rate_limit(self):
        """简单速率限制"""
        import time
        elapsed = time.time() - self._last_request_time
        if elapsed < self._min_interval:
            await asyncio.sleep(self._min_interval - elapsed)
        self._last_request_time = time.time()
    
    async def _request(self, endpoint: str, params: Optional[Dict] = None) -> Dict:
        """发送API请求"""
        await self._rate_limit()
        
        session = await self._get_session()
        url = f"{self.base_url}{endpoint}"
        
        try:
            async with session.get(url, params=params) as response:
                if response.status == 429:
                    logger.warning("CoinMetrics API速率限制，等待60秒...")
                    await asyncio.sleep(60)
                    return await self._request(endpoint, params)
                
                response.raise_for_status()
                return await response.json()
                
        except aiohttp.ClientError as e:
            logger.error(f"CoinMetrics API请求失败: {e}")
            raise
    
    async def get_network_metrics(
        self,
        asset: str = "btc",
        metrics: Optional[List[str]] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        frequency: str = "1d"
    ) -> pd.DataFrame:
        """
        获取链上网络指标
        
        Args:
            asset: 资产代码 (btc, eth等)
            metrics: 要获取的指标列表，默认获取常用指标
            start_time: 开始时间
            end_time: 结束时间
            frequency: 数据频率 (1d=日级)
            
        Returns:
            DataFrame with timestamp index and metric columns
        """
        if metrics is None:
            # 默认获取关键链上指标
            metrics = [
                "AdrActCnt",      # 活跃地址
                "TxCnt",          # 交易数
                "HashRate",       # 哈希率
                "FeeMeanUSD",     # 平均手续费
                "NVTAdj",         # NVT比率
                "SplyCur",        # 当前供应
            ]
        
        if end_time is None:
            end_time = datetime.utcnow()
        if start_time is None:
            start_time = end_time - timedelta(days=90)
        
        logger.info(f"📥 从CoinMetrics获取 {asset.upper()} 链上数据...")
        logger.info(f"   指标: {', '.join(metrics)}")
        
        try:
            params = {
                "assets": asset,
                "metrics": ",".join(metrics),
                "start_time": start_time.strftime("%Y-%m-%dT%H:%M:%S"),
                "end_time": end_time.strftime("%Y-%m-%dT%H:%M:%S"),
                "frequency": frequency,
                "page_size": 10000
            }
            
            data = await self._request("/timeseries/asset-metrics", params)
            
            if not data or "data" not in data:
                logger.warning("CoinMetrics返回空数据")
                return pd.DataFrame()
            
            # 解析数据
            records = []
            for item in data["data"]:
                record = {"timestamp": pd.to_datetime(item["time"])}
                for metric in metrics:
                    value = item.get(metric)
                    if value is not None:
                        try:
                            record[f"cm_{metric}"] = float(value)
                        except (ValueError, TypeError):
                            record[f"cm_{metric}"] = None
                records.append(record)
            
            df = pd.DataFrame(records)
            if not df.empty:
                df.set_index("timestamp", inplace=True)
                df.sort_index(inplace=True)
                logger.info(f"✅ CoinMetrics数据: {len(df)} 条记录")
                logger.info(f"   时间范围: {df.index.min()} ~ {df.index.max()}")
            
            return df
            
        except Exception as e:
            logger.error(f"CoinMetrics数据获取失败: {e}")
            return pd.DataFrame()
    
    async def get_exchange_flows(
        self,
        asset: str = "btc",
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        获取交易所流入流出数据
        
        Args:
            asset: 资产代码
            start_time: 开始时间
            end_time: 结束时间
            
        Returns:
            DataFrame with exchange flow metrics
        """
        flow_metrics = [
            "FlowInExNtv",      # 流入交易所(原生单位)
            "FlowOutExNtv",     # 流出交易所(原生单位)
            "FlowInExUSD",      # 流入交易所(USD)
            "FlowOutExUSD",     # 流出交易所(USD)
        ]
        
        return await self.get_network_metrics(
            asset=asset,
            metrics=flow_metrics,
            start_time=start_time,
            end_time=end_time
        )
    
    async def get_miner_metrics(
        self,
        asset: str = "btc",
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        获取矿工相关指标
        
        Args:
            asset: 资产代码
            start_time: 开始时间
            end_time: 结束时间
            
        Returns:
            DataFrame with miner metrics
        """
        miner_metrics = [
            "HashRate",         # 哈希率
            "DiffMean",         # 难度
            "RevHashRateUSD",   # 每单位算力收益
            "BlkCnt",           # 区块数
        ]
        
        return await self.get_network_metrics(
            asset=asset,
            metrics=miner_metrics,
            start_time=start_time,
            end_time=end_time
        )
    
    async def get_hourly_ohlcv(self, symbol: str, days: int = 90, vs_currency: str = "usd"):
        """实现基类接口 - CoinMetrics主要提供日级数据"""
        raise NotImplementedError("CoinMetrics主要提供日级链上数据，请使用get_network_metrics()")
    
    async def get_current_price(self, symbol: str) -> float:
        """获取当前价格"""
        try:
            params = {
                "assets": symbol.lower(),
                "metrics": "PriceUSD"
            }
            data = await self._request("/timeseries/asset-metrics", params)
            if data and "data" in data and len(data["data"]) > 0:
                return float(data["data"][-1]["PriceUSD"])
        except Exception as e:
            logger.error(f"获取价格失败: {e}")
        return 0.0
    
    def get_rate_limit(self) -> RateLimitInfo:
        """获取速率限制信息"""
        return RateLimitInfo(
            calls_per_minute=60,
            daily_limit=None  # 社区版无明确日限制
        )
    
    async def health_check(self) -> bool:
        """检查API连接状态"""
        try:
            data = await self._request("/catalog/assets")
            return data is not None
        except Exception:
            return False
    
    async def close(self):
        """关闭HTTP会话"""
        if self._session and not self._session.closed:
            await self._session.close()
