"""
Financial Modeling Prep (FMP) 数据获取器
================================
实现FMP API的BTC数据获取，支持更长历史数据

FMP API特点:
- 免费版: 250请求/天
- 支持加密货币历史数据
- 提供1分钟、5分钟、15分钟、30分钟、1小时、4小时、日线数据
"""

import asyncio
import aiohttp
import logging
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
import time
import pandas as pd

from .base import (
    DataFetcher, 
    DataSource, 
    OHLCV, 
    MarketData, 
    RateLimitInfo,
    DataFetcherFactory
)
from .cache import cache_manager

logger = logging.getLogger(__name__)


class FMPRateLimiter:
    """FMP速率限制器 - 250请求/天"""
    
    def __init__(self, calls_per_minute: int = 5):
        self.calls_per_minute = calls_per_minute
        self.min_interval = 60.0 / calls_per_minute
        self.last_call_time = 0.0
        self._lock = asyncio.Lock()
        self.daily_calls = 0
        self.daily_limit = 250
        self.last_reset_date = datetime.now().date()
    
    async def acquire(self):
        """获取请求许可"""
        async with self._lock:
            # 检查日期是否变化，重置每日计数
            current_date = datetime.now().date()
            if current_date != self.last_reset_date:
                self.daily_calls = 0
                self.last_reset_date = current_date
            
            # 检查每日限制
            if self.daily_calls >= self.daily_limit:
                raise Exception(f"FMP API 每日请求限制已达到 ({self.daily_limit})")
            
            current_time = time.time()
            elapsed = current_time - self.last_call_time
            wait_time = self.min_interval - elapsed
            
            if wait_time > 0:
                logger.debug(f"FMP速率限制: 等待 {wait_time:.2f} 秒")
                await asyncio.sleep(wait_time)
            
            self.last_call_time = time.time()
            self.daily_calls += 1


class FMPFetcher(DataFetcher):
    """
    Financial Modeling Prep 数据获取器
    
    特点:
    - 支持BTC/USD历史数据
    - 提供多种时间粒度
    - 免费版每天250次请求
    """
    
    def __init__(self, api_key: str = ""):
        # 注册 FMP 数据源（如果尚未在枚举中）
        super().__init__(DataSource.FMP if hasattr(DataSource, 'FMP') else DataSource.COINGECKO)
        self.api_key = api_key
        self.base_url = "https://financialmodelingprep.com/api/v3"
        self.rate_limiter = FMPRateLimiter()
        self._session: Optional[aiohttp.ClientSession] = None
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """获取或创建HTTP会话"""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                headers={
                    "Accept": "application/json",
                    "User-Agent": "btc-predictor/1.0"
                }
            )
        return self._session
    
    async def _request(self, endpoint: str, params: Optional[Dict] = None) -> Any:
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
        
        # 添加API密钥
        if params is None:
            params = {}
        params['apikey'] = self.api_key
        
        try:
            async with session.get(url, params=params) as response:
                cache_manager.record_api_request(
                    "fmp", 
                    endpoint, 
                    response.status == 200
                )
                
                if response.status == 429:
                    logger.warning("FMP API速率限制，等待60秒...")
                    await asyncio.sleep(60)
                    return await self._request(endpoint, params)
                
                if response.status == 401:
                    raise Exception("FMP API密钥无效或未提供")
                
                response.raise_for_status()
                return await response.json()
                
        except aiohttp.ClientError as e:
            logger.error(f"FMP API请求失败: {e}")
            raise
    
    async def get_hourly_ohlcv(
        self, 
        symbol: str = "BTCUSD",
        days: int = 90,
        vs_currency: str = "usd"
    ) -> MarketData:
        """
        获取小时级OHLCV数据
        
        FMP 历史数据端点:
        - /historical-chart/1hour/{symbol}: 1小时K线
        - /historical-chart/4hour/{symbol}: 4小时K线
        - /historical-chart/1day/{symbol}: 日K线
        
        Args:
            symbol: 交易对符号 (如 "BTCUSD")
            days: 历史天数
            vs_currency: 计价货币（用于兼容，FMP固定为USD）
            
        Returns:
            MarketData对象
        """
        if not self.api_key:
            raise ValueError("FMP API密钥未设置，请在 .env 文件中设置 FMP_API_KEY")
        
        # FMP使用BTCUSD格式
        fmp_symbol = "BTCUSD"
        
        logger.info(f"📥 从FMP获取 {fmp_symbol} {days}天小时数据...")
        
        # 先检查缓存
        cached_df = cache_manager.get_ohlcv(
            "bitcoin", 
            "fmp",
            start_time=datetime.now() - timedelta(days=days)
        )
        
        if not cached_df.empty:
            latest_cached = cached_df.index.max()
            if datetime.now() - latest_cached.to_pydatetime() < timedelta(hours=1):
                logger.info(f"使用FMP缓存数据: {len(cached_df)} 条记录")
                return self._df_to_market_data("bitcoin", cached_df)
        
        # 获取小时数据
        try:
            # FMP返回最近的数据，需要指定日期范围
            from_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
            to_date = datetime.now().strftime("%Y-%m-%d")
            
            data = await self._request(
                f"/historical-chart/1hour/{fmp_symbol}",
                params={
                    "from": from_date,
                    "to": to_date
                }
            )
            
            if not data:
                logger.warning("FMP返回空数据")
                return MarketData(symbol="bitcoin", ohlcv_data=[])
            
            # 转换为OHLCV对象列表
            ohlcv_list = []
            for item in data:
                try:
                    # FMP日期格式: "2024-01-15 10:00:00"
                    timestamp = datetime.strptime(item['date'], "%Y-%m-%d %H:%M:%S")
                    ohlcv = OHLCV(
                        timestamp=timestamp,
                        open=float(item['open']),
                        high=float(item['high']),
                        low=float(item['low']),
                        close=float(item['close']),
                        volume=float(item.get('volume', 0))
                    )
                    ohlcv_list.append(ohlcv)
                except (KeyError, ValueError) as e:
                    logger.warning(f"跳过无效数据行: {e}")
                    continue
            
            # 按时间排序（FMP返回的是降序）
            ohlcv_list.sort(key=lambda x: x.timestamp)
            
            logger.info(f"✅ FMP数据获取成功: {len(ohlcv_list)} 条记录")
            if ohlcv_list:
                logger.info(f"   时间范围: {ohlcv_list[0].timestamp} ~ {ohlcv_list[-1].timestamp}")
            
            # 创建MarketData
            market_data = MarketData(
                symbol="bitcoin",
                ohlcv_data=ohlcv_list
            )
            
            # 保存到缓存
            df = market_data.to_dataframe()
            if not df.empty:
                cache_manager.save_ohlcv("bitcoin", "fmp", df)
            
            return market_data
            
        except Exception as e:
            logger.error(f"FMP数据获取失败: {e}")
            raise
    
    async def get_daily_ohlcv(
        self, 
        symbol: str = "BTCUSD",
        days: int = 365
    ) -> MarketData:
        """
        获取日级OHLCV数据（支持更长历史）
        
        Args:
            symbol: 交易对符号
            days: 历史天数
            
        Returns:
            MarketData对象
        """
        if not self.api_key:
            raise ValueError("FMP API密钥未设置")
        
        fmp_symbol = "BTCUSD"
        logger.info(f"📥 从FMP获取 {fmp_symbol} {days}天日线数据...")
        
        try:
            from_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
            to_date = datetime.now().strftime("%Y-%m-%d")
            
            data = await self._request(
                f"/historical-price-full/{fmp_symbol}",
                params={
                    "from": from_date,
                    "to": to_date
                }
            )
            
            if not data or 'historical' not in data:
                logger.warning("FMP返回空数据")
                return MarketData(symbol="bitcoin", ohlcv_data=[])
            
            ohlcv_list = []
            for item in data['historical']:
                try:
                    timestamp = datetime.strptime(item['date'], "%Y-%m-%d")
                    ohlcv = OHLCV(
                        timestamp=timestamp,
                        open=float(item['open']),
                        high=float(item['high']),
                        low=float(item['low']),
                        close=float(item['close']),
                        volume=float(item.get('volume', 0))
                    )
                    ohlcv_list.append(ohlcv)
                except (KeyError, ValueError) as e:
                    continue
            
            ohlcv_list.sort(key=lambda x: x.timestamp)
            
            logger.info(f"✅ FMP日线数据: {len(ohlcv_list)} 条记录")
            
            return MarketData(
                symbol="bitcoin",
                ohlcv_data=ohlcv_list
            )
            
        except Exception as e:
            logger.error(f"FMP日线数据获取失败: {e}")
            raise
    
    async def get_current_price(self, symbol: str = "BTCUSD") -> float:
        """获取当前价格"""
        if not self.api_key:
            raise ValueError("FMP API密钥未设置")
        
        try:
            data = await self._request(f"/quote/{symbol}")
            if data and len(data) > 0:
                return float(data[0]['price'])
            raise Exception("无法获取价格数据")
        except Exception as e:
            logger.error(f"获取FMP当前价格失败: {e}")
            raise
    
    def get_rate_limit(self) -> RateLimitInfo:
        """获取速率限制信息"""
        return RateLimitInfo(
            calls_per_minute=5,
            daily_limit=250,
            remaining_calls=250 - self.rate_limiter.daily_calls
        )
    
    async def health_check(self) -> bool:
        """检查API连接状态"""
        if not self.api_key:
            return False
        try:
            data = await self._request("/quote/BTCUSD")
            return data is not None and len(data) > 0
        except Exception:
            return False
    
    async def close(self):
        """关闭HTTP会话"""
        if self._session and not self._session.closed:
            await self._session.close()
    
    def _df_to_market_data(self, symbol: str, df: pd.DataFrame) -> MarketData:
        """DataFrame转换为MarketData"""
        from datetime import datetime
        ohlcv_list = []
        for idx, row in df.iterrows():
            # 转换索引为datetime
            if isinstance(idx, pd.Timestamp):
                ts = idx.to_pydatetime()
            elif isinstance(idx, datetime):
                ts = idx
            else:
                ts = pd.Timestamp(str(idx)).to_pydatetime()
            ohlcv = OHLCV(
                timestamp=ts,
                open=float(row['open']),
                high=float(row['high']),
                low=float(row['low']),
                close=float(row['close']),
                volume=float(row.get('volume', 0))
            )
            ohlcv_list.append(ohlcv)
        
        return MarketData(symbol=symbol, ohlcv_data=ohlcv_list)
    
    # ==================== 宏观经济数据 ====================
    
    async def get_economic_indicators(
        self,
        indicator: str = "GDP",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        获取宏观经济指标
        
        支持的指标:
        - GDP: 美国GDP
        - realGDP: 实际GDP
        - CPI: 消费者价格指数
        - inflationRate: 通胀率
        - interestRate: 利率
        - unemployment: 失业率
        - retailSales: 零售销售
        - durableGoods: 耐用品订单
        - industrialProduction: 工业生产指数
        - consumerSentiment: 消费者信心指数
        
        Args:
            indicator: 指标名称
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
            
        Returns:
            DataFrame with economic data
        """
        if not self.api_key:
            raise ValueError("FMP API密钥未设置")
        
        logger.info(f"📥 获取宏观指标: {indicator}")
        
        try:
            params = {}
            if start_date:
                params["from"] = start_date
            if end_date:
                params["to"] = end_date
            
            data = await self._request(f"/economic", params={"name": indicator, **params})
            
            if not data:
                return pd.DataFrame()
            
            df = pd.DataFrame(data)
            if "date" in df.columns:
                df["timestamp"] = pd.to_datetime(df["date"])
                df.set_index("timestamp", inplace=True)
                df.sort_index(inplace=True)
            
            logger.info(f"✅ {indicator} 数据: {len(df)} 条记录")
            return df
            
        except Exception as e:
            logger.error(f"获取宏观指标失败: {e}")
            return pd.DataFrame()
    
    async def get_treasury_rates(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        获取美国国债收益率
        
        包含: 1M, 2M, 3M, 6M, 1Y, 2Y, 3Y, 5Y, 7Y, 10Y, 20Y, 30Y
        
        Returns:
            DataFrame with treasury rates
        """
        if not self.api_key:
            raise ValueError("FMP API密钥未设置")
        
        logger.info("📥 获取美国国债收益率...")
        
        try:
            params = {}
            if start_date:
                params["from"] = start_date
            if end_date:
                params["to"] = end_date
            
            data = await self._request("/treasury", params=params)
            
            if not data:
                return pd.DataFrame()
            
            df = pd.DataFrame(data)
            if "date" in df.columns:
                df["timestamp"] = pd.to_datetime(df["date"])
                df.set_index("timestamp", inplace=True)
                df.sort_index(inplace=True)
                
                # 重命名列添加前缀
                rename_cols = {col: f"treasury_{col}" for col in df.columns if col != "date"}
                df.rename(columns=rename_cols, inplace=True)
            
            logger.info(f"✅ 国债收益率数据: {len(df)} 条记录")
            return df
            
        except Exception as e:
            logger.error(f"获取国债收益率失败: {e}")
            return pd.DataFrame()
    
    # ==================== 股票指数数据 ====================
    
    async def get_index_data(
        self,
        symbol: str = "^GSPC",  # S&P 500
        days: int = 90
    ) -> pd.DataFrame:
        """
        获取股票指数历史数据
        
        常用符号:
        - ^GSPC: S&P 500
        - ^DJI: 道琼斯工业平均
        - ^IXIC: 纳斯达克综合
        - ^VIX: VIX恐慌指数
        - ^TNX: 10年期国债收益率
        
        Args:
            symbol: 指数符号
            days: 历史天数
            
        Returns:
            DataFrame with OHLCV data
        """
        if not self.api_key:
            raise ValueError("FMP API密钥未设置")
        
        logger.info(f"📥 获取指数数据: {symbol}")
        
        try:
            from_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
            to_date = datetime.now().strftime("%Y-%m-%d")
            
            data = await self._request(
                f"/historical-price-full/{symbol}",
                params={"from": from_date, "to": to_date}
            )
            
            if not data or "historical" not in data:
                return pd.DataFrame()
            
            df = pd.DataFrame(data["historical"])
            if "date" in df.columns:
                df["timestamp"] = pd.to_datetime(df["date"])
                df.set_index("timestamp", inplace=True)
                df.sort_index(inplace=True)
                
                # 只保留需要的列并重命名
                symbol_prefix = symbol.replace("^", "").lower()
                df = df[["open", "high", "low", "close", "volume"]].copy()
                df.columns = [f"{symbol_prefix}_{col}" for col in df.columns]
            
            logger.info(f"✅ {symbol} 数据: {len(df)} 条记录")
            return df
            
        except Exception as e:
            logger.error(f"获取指数数据失败: {e}")
            return pd.DataFrame()
    
    async def get_fear_greed_index(self) -> pd.DataFrame:
        """
        获取市场恐惧贪婪指数
        
        Returns:
            DataFrame with fear & greed data
        """
        if not self.api_key:
            raise ValueError("FMP API密钥未设置")
        
        logger.info("📥 获取恐惧贪婪指数...")
        
        try:
            data = await self._request("/fear-and-greed-index")
            
            if not data:
                return pd.DataFrame()
            
            df = pd.DataFrame(data)
            if "date" in df.columns:
                df["timestamp"] = pd.to_datetime(df["date"])
                df.set_index("timestamp", inplace=True)
                df.sort_index(inplace=True)
            
            logger.info(f"✅ 恐惧贪婪指数: {len(df)} 条记录")
            return df
            
        except Exception as e:
            logger.error(f"获取恐惧贪婪指数失败: {e}")
            return pd.DataFrame()
    
    # ==================== 商品数据 ====================
    
    async def get_commodity_data(
        self,
        symbol: str = "GCUSD",  # 黄金
        days: int = 90
    ) -> pd.DataFrame:
        """
        获取商品历史数据
        
        常用符号:
        - GCUSD: 黄金
        - SIUSD: 白银
        - CLUSD: 原油(WTI)
        - NGUSD: 天然气
        - HGUSD: 铜
        
        Args:
            symbol: 商品符号
            days: 历史天数
            
        Returns:
            DataFrame with OHLCV data
        """
        if not self.api_key:
            raise ValueError("FMP API密钥未设置")
        
        logger.info(f"📥 获取商品数据: {symbol}")
        
        try:
            from_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
            to_date = datetime.now().strftime("%Y-%m-%d")
            
            data = await self._request(
                f"/historical-price-full/{symbol}",
                params={"from": from_date, "to": to_date}
            )
            
            if not data or "historical" not in data:
                return pd.DataFrame()
            
            df = pd.DataFrame(data["historical"])
            if "date" in df.columns:
                df["timestamp"] = pd.to_datetime(df["date"])
                df.set_index("timestamp", inplace=True)
                df.sort_index(inplace=True)
                
                # 只保留需要的列并重命名
                symbol_prefix = symbol.lower().replace("usd", "")
                df = df[["open", "high", "low", "close", "volume"]].copy()
                df.columns = [f"{symbol_prefix}_{col}" for col in df.columns]
            
            logger.info(f"✅ {symbol} 数据: {len(df)} 条记录")
            return df
            
        except Exception as e:
            logger.error(f"获取商品数据失败: {e}")
            return pd.DataFrame()
    
    # ==================== 外汇数据 ====================
    
    async def get_forex_data(
        self,
        symbol: str = "EURUSD",
        days: int = 90
    ) -> pd.DataFrame:
        """
        获取外汇历史数据
        
        常用货币对:
        - EURUSD: 欧元/美元
        - GBPUSD: 英镑/美元
        - USDJPY: 美元/日元
        - USDCNY: 美元/人民币
        - DXY: 美元指数
        
        Args:
            symbol: 货币对符号
            days: 历史天数
            
        Returns:
            DataFrame with OHLCV data
        """
        if not self.api_key:
            raise ValueError("FMP API密钥未设置")
        
        logger.info(f"📥 获取外汇数据: {symbol}")
        
        try:
            from_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
            to_date = datetime.now().strftime("%Y-%m-%d")
            
            data = await self._request(
                f"/historical-price-full/{symbol}",
                params={"from": from_date, "to": to_date}
            )
            
            if not data or "historical" not in data:
                return pd.DataFrame()
            
            df = pd.DataFrame(data["historical"])
            if "date" in df.columns:
                df["timestamp"] = pd.to_datetime(df["date"])
                df.set_index("timestamp", inplace=True)
                df.sort_index(inplace=True)
                
                # 只保留需要的列并重命名
                symbol_prefix = symbol.lower()
                df = df[["open", "high", "low", "close", "volume"]].copy()
                df.columns = [f"{symbol_prefix}_{col}" for col in df.columns]
            
            logger.info(f"✅ {symbol} 数据: {len(df)} 条记录")
            return df
            
        except Exception as e:
            logger.error(f"获取外汇数据失败: {e}")
            return pd.DataFrame()
    
    # ==================== 新闻数据 ====================
    
    async def get_crypto_news(
        self,
        symbol: str = "BTCUSD",
        limit: int = 50
    ) -> pd.DataFrame:
        """
        获取加密货币新闻
        
        Args:
            symbol: 交易对符号
            limit: 新闻数量限制
            
        Returns:
            DataFrame with news data
        """
        if not self.api_key:
            raise ValueError("FMP API密钥未设置")
        
        logger.info(f"📥 获取加密新闻: {symbol}")
        
        try:
            data = await self._request(
                "/stock_news",
                params={"tickers": symbol, "limit": limit}
            )
            
            if not data:
                return pd.DataFrame()
            
            df = pd.DataFrame(data)
            if "publishedDate" in df.columns:
                df["timestamp"] = pd.to_datetime(df["publishedDate"])
                df.set_index("timestamp", inplace=True)
                df.sort_index(inplace=True)
            
            logger.info(f"✅ 新闻数据: {len(df)} 条")
            return df
            
        except Exception as e:
            logger.error(f"获取新闻失败: {e}")
            return pd.DataFrame()
    
    async def get_general_news(self, limit: int = 50) -> pd.DataFrame:
        """
        获取综合财经新闻
        
        Args:
            limit: 新闻数量限制
            
        Returns:
            DataFrame with news data
        """
        if not self.api_key:
            raise ValueError("FMP API密钥未设置")
        
        logger.info("📥 获取综合财经新闻...")
        
        try:
            data = await self._request("/fmp/articles", params={"page": 0, "size": limit})
            
            if not data or "content" not in data:
                return pd.DataFrame()
            
            df = pd.DataFrame(data["content"])
            if "date" in df.columns:
                df["timestamp"] = pd.to_datetime(df["date"])
                df.set_index("timestamp", inplace=True)
                df.sort_index(inplace=True)
            
            logger.info(f"✅ 综合新闻: {len(df)} 条")
            return df
            
        except Exception as e:
            logger.error(f"获取综合新闻失败: {e}")
            return pd.DataFrame()

