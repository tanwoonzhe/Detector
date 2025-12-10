"""
Binance 历史数据归档下载器
================================
从 Binance Data Vision 下载完整历史K线数据

数据源: https://data.binance.vision/
支持粒度: 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d

特点:
- 完整历史数据 (2017年至今)
- 官方数据源，质量高
- 支持批量下载和本地缓存
"""

import asyncio
import aiohttp
import logging
import zipfile
import io
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Tuple
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Binance Data Vision 基础URL
BASE_URL = "https://data.binance.vision/data/spot"

# 支持的时间粒度
VALID_INTERVALS = ["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d"]


class BinanceHistoricalFetcher:
    """
    Binance 历史数据下载器
    
    从 Binance Data Vision 下载月度/日度归档K线数据
    """
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """
        初始化
        
        Args:
            cache_dir: 缓存目录，默认为 data/raw/binance_historical/
        """
        if cache_dir is None:
            cache_dir = Path(__file__).parent.parent.parent / "data" / "raw" / "binance_historical"
        
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._session: Optional[aiohttp.ClientSession] = None
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """获取HTTP会话"""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=300)  # 5分钟超时
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session
    
    async def close(self):
        """关闭会话"""
        if self._session and not self._session.closed:
            await self._session.close()
    
    def _get_monthly_url(self, symbol: str, interval: str, year: int, month: int) -> str:
        """生成月度数据URL"""
        return f"{BASE_URL}/monthly/klines/{symbol}/{interval}/{symbol}-{interval}-{year}-{month:02d}.zip"
    
    def _get_daily_url(self, symbol: str, interval: str, date: datetime) -> str:
        """生成日度数据URL"""
        date_str = date.strftime("%Y-%m-%d")
        return f"{BASE_URL}/daily/klines/{symbol}/{interval}/{symbol}-{interval}-{date_str}.zip"
    
    async def _download_zip(self, url: str) -> Optional[bytes]:
        """下载ZIP文件"""
        session = await self._get_session()
        
        try:
            async with session.get(url) as response:
                if response.status == 200:
                    return await response.read()
                elif response.status == 404:
                    return None  # 文件不存在
                else:
                    logger.warning(f"下载失败 {url}: HTTP {response.status}")
                    return None
        except Exception as e:
            logger.warning(f"下载异常 {url}: {e}")
            return None
    
    def _parse_kline_csv(self, csv_content: str) -> pd.DataFrame:
        """解析K线CSV数据"""
        from io import StringIO
        
        # Binance K线列: Open time, Open, High, Low, Close, Volume, Close time, 
        # Quote volume, Number of trades, Taker buy base, Taker buy quote, Ignore
        columns = [
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades", "taker_buy_base",
            "taker_buy_quote", "ignore"
        ]
        
        df = pd.read_csv(StringIO(csv_content), names=columns, header=None)
        
        # 转换时间戳
        df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms")
        
        # 转换数值类型
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        
        # 只保留需要的列
        df = df[["timestamp", "open", "high", "low", "close", "volume"]].copy()
        df.set_index("timestamp", inplace=True)
        
        return df
    
    async def download_monthly_data(
        self,
        symbol: str = "BTCUSDT",
        interval: str = "1h",
        start_year: int = 2017,
        start_month: int = 8,
        end_year: Optional[int] = None,
        end_month: Optional[int] = None,
        show_progress: bool = True
    ) -> pd.DataFrame:
        """
        下载月度K线数据
        
        Args:
            symbol: 交易对 (如 BTCUSDT)
            interval: 时间粒度 (1m, 5m, 15m, 30m, 1h, 4h, 1d)
            start_year: 开始年份
            start_month: 开始月份
            end_year: 结束年份 (默认当前年)
            end_month: 结束月份 (默认当前月)
            show_progress: 显示进度条
            
        Returns:
            合并后的DataFrame
        """
        if interval not in VALID_INTERVALS:
            raise ValueError(f"无效的时间粒度: {interval}，支持: {VALID_INTERVALS}")
        
        if end_year is None:
            end_year = datetime.now().year
        if end_month is None:
            end_month = datetime.now().month
        
        # 生成月份列表
        months: List[Tuple[int, int]] = []
        year, month = start_year, start_month
        while (year, month) <= (end_year, end_month):
            months.append((year, month))
            month += 1
            if month > 12:
                month = 1
                year += 1
        
        logger.info(f"📥 下载 {symbol} {interval} 数据: {start_year}-{start_month:02d} ~ {end_year}-{end_month:02d}")
        logger.info(f"   共 {len(months)} 个月")
        
        # 检查缓存
        cache_file = self.cache_dir / f"{symbol}_{interval}_monthly.parquet"
        
        all_dfs = []
        
        # 下载进度条
        iterator = tqdm(months, desc=f"下载 {symbol} {interval}") if show_progress else months
        
        for year, month in iterator:
            # 检查本地缓存
            month_cache = self.cache_dir / f"{symbol}_{interval}_{year}-{month:02d}.csv"
            
            if month_cache.exists():
                df = pd.read_csv(month_cache, parse_dates=["timestamp"], index_col="timestamp")
                all_dfs.append(df)
                continue
            
            # 下载
            url = self._get_monthly_url(symbol, interval, year, month)
            zip_data = await self._download_zip(url)
            
            if zip_data is None:
                continue
            
            # 解压并解析
            try:
                with zipfile.ZipFile(io.BytesIO(zip_data)) as zf:
                    for name in zf.namelist():
                        if name.endswith(".csv"):
                            csv_content = zf.read(name).decode("utf-8")
                            df = self._parse_kline_csv(csv_content)
                            
                            # 保存到本地缓存
                            df.to_csv(month_cache)
                            all_dfs.append(df)
                            break
            except Exception as e:
                logger.warning(f"解析失败 {year}-{month:02d}: {e}")
        
        if not all_dfs:
            logger.warning("没有下载到任何数据")
            return pd.DataFrame()
        
        # 合并所有数据
        df_merged = pd.concat(all_dfs)
        df_merged.sort_index(inplace=True)
        df_merged = df_merged[~df_merged.index.duplicated(keep="first")]
        
        # 保存合并后的parquet
        df_merged.to_parquet(cache_file)
        
        logger.info(f"✅ 下载完成: {len(df_merged)} 条记录")
        logger.info(f"   时间范围: {df_merged.index.min()} ~ {df_merged.index.max()}")
        logger.info(f"   缓存位置: {cache_file}")
        
        return df_merged
    
    def load_cached_data(
        self,
        symbol: str = "BTCUSDT",
        interval: str = "1h"
    ) -> pd.DataFrame:
        """
        加载本地缓存数据
        
        Args:
            symbol: 交易对
            interval: 时间粒度
            
        Returns:
            DataFrame
        """
        cache_file = self.cache_dir / f"{symbol}_{interval}_monthly.parquet"
        
        if cache_file.exists():
            logger.info(f"📁 加载缓存: {cache_file}")
            df = pd.read_parquet(cache_file)
            logger.info(f"   {len(df)} 条记录, {df.index.min()} ~ {df.index.max()}")
            return df
        
        # 尝试合并月度CSV
        csv_files = sorted(self.cache_dir.glob(f"{symbol}_{interval}_*.csv"))
        if csv_files:
            logger.info(f"📁 合并 {len(csv_files)} 个月度文件...")
            dfs = []
            for f in csv_files:
                df = pd.read_csv(f, parse_dates=["timestamp"], index_col="timestamp")
                dfs.append(df)
            
            if dfs:
                df_merged = pd.concat(dfs)
                df_merged.sort_index(inplace=True)
                df_merged = df_merged[~df_merged.index.duplicated(keep="first")]
                df_merged.to_parquet(cache_file)
                return df_merged
        
        logger.warning(f"未找到缓存数据: {symbol} {interval}")
        return pd.DataFrame()
    
    async def download_recent_daily(
        self,
        symbol: str = "BTCUSDT",
        interval: str = "1h",
        days: int = 30
    ) -> pd.DataFrame:
        """
        下载最近N天的日度数据（用于补充最新数据）
        
        Args:
            symbol: 交易对
            interval: 时间粒度
            days: 天数
            
        Returns:
            DataFrame
        """
        logger.info(f"📥 下载 {symbol} {interval} 最近 {days} 天数据...")
        
        all_dfs = []
        end_date = datetime.now()
        
        for i in range(days):
            date = end_date - timedelta(days=i)
            url = self._get_daily_url(symbol, interval, date)
            zip_data = await self._download_zip(url)
            
            if zip_data is None:
                continue
            
            try:
                with zipfile.ZipFile(io.BytesIO(zip_data)) as zf:
                    for name in zf.namelist():
                        if name.endswith(".csv"):
                            csv_content = zf.read(name).decode("utf-8")
                            df = self._parse_kline_csv(csv_content)
                            all_dfs.append(df)
                            break
            except Exception as e:
                logger.warning(f"解析失败 {date.date()}: {e}")
        
        if not all_dfs:
            return pd.DataFrame()
        
        df_merged = pd.concat(all_dfs)
        df_merged.sort_index(inplace=True)
        df_merged = df_merged[~df_merged.index.duplicated(keep="first")]
        
        logger.info(f"✅ 下载完成: {len(df_merged)} 条记录")
        
        return df_merged


async def download_btc_historical(
    interval: str = "1h",
    start_year: int = 2017,
    show_progress: bool = True
) -> pd.DataFrame:
    """
    便捷函数：下载BTC历史数据
    
    Args:
        interval: 时间粒度 (1m, 5m, 15m, 30m, 1h, 4h, 1d)
        start_year: 开始年份
        show_progress: 显示进度条
        
    Returns:
        DataFrame with OHLCV data
    """
    fetcher = BinanceHistoricalFetcher()
    
    try:
        df = await fetcher.download_monthly_data(
            symbol="BTCUSDT",
            interval=interval,
            start_year=start_year,
            start_month=8 if start_year == 2017 else 1,
            show_progress=show_progress
        )
        return df
    finally:
        await fetcher.close()


def load_btc_historical(interval: str = "1h") -> pd.DataFrame:
    """
    便捷函数：加载本地缓存的BTC历史数据
    
    Args:
        interval: 时间粒度
        
    Returns:
        DataFrame
    """
    fetcher = BinanceHistoricalFetcher()
    return fetcher.load_cached_data(symbol="BTCUSDT", interval=interval)
