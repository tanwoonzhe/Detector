"""
多数据源合并管道
================================
统一获取、合并、对齐来自多个数据源的数据

支持的数据源:
- CoinGecko: 加密货币价格/市场数据
- FMP: 宏观经济/股票指数/商品/外汇/新闻
- CoinMetrics: 链上数据/网络指标
- HuggingFace: 历史数据集

使用方法:
    pipeline = DataPipeline(fmp_api_key="your_key")
    df = await pipeline.fetch_all(days=90)
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any
import pandas as pd
import numpy as np

from .coingecko_fetcher import CoinGeckoFetcher
from .fmp_fetcher import FMPFetcher
from .coinmetrics_fetcher import CoinMetricsFetcher

logger = logging.getLogger(__name__)


class DataPipeline:
    """
    多数据源合并管道
    
    统一管理多个数据源的获取和合并，生成用于训练的特征矩阵
    """
    
    def __init__(
        self,
        fmp_api_key: str = "",
        coinmetrics_api_key: str = "",
        use_cache: bool = True
    ):
        """
        初始化数据管道
        
        Args:
            fmp_api_key: FMP API密钥
            coinmetrics_api_key: CoinMetrics API密钥（社区版可为空）
            use_cache: 是否使用缓存
        """
        self.fmp_api_key = fmp_api_key
        self.coinmetrics_api_key = coinmetrics_api_key
        self.use_cache = use_cache
        
        # 初始化各数据源
        self._coingecko: Optional[CoinGeckoFetcher] = None
        self._fmp: Optional[FMPFetcher] = None
        self._coinmetrics: Optional[CoinMetricsFetcher] = None
    
    @property
    def coingecko(self) -> CoinGeckoFetcher:
        if self._coingecko is None:
            self._coingecko = CoinGeckoFetcher()
        return self._coingecko
    
    @property
    def fmp(self) -> FMPFetcher:
        if self._fmp is None:
            self._fmp = FMPFetcher(api_key=self.fmp_api_key)
        return self._fmp
    
    @property
    def coinmetrics(self) -> CoinMetricsFetcher:
        if self._coinmetrics is None:
            self._coinmetrics = CoinMetricsFetcher(api_key=self.coinmetrics_api_key)
        return self._coinmetrics
    
    async def close(self):
        """关闭所有连接"""
        if self._coingecko:
            await self._coingecko.close()
        if self._fmp:
            await self._fmp.close()
        if self._coinmetrics:
            await self._coinmetrics.close()
    
    # ==================== 数据获取方法 ====================
    
    async def fetch_btc_price(self, days: int = 90) -> pd.DataFrame:
        """
        获取BTC价格数据 (OHLCV)
        
        优先使用CoinGecko，失败则尝试FMP
        """
        logger.info("📊 获取BTC价格数据...")
        
        try:
            market_data = await self.coingecko.get_hourly_ohlcv(
                symbol="bitcoin",
                days=days
            )
            df = market_data.to_dataframe()
            if not df.empty:
                logger.info(f"✅ CoinGecko价格数据: {len(df)} 条")
                return df
        except Exception as e:
            logger.warning(f"CoinGecko获取失败: {e}")
        
        # 回退到FMP
        if self.fmp_api_key:
            try:
                market_data = await self.fmp.get_hourly_ohlcv(
                    symbol="BTCUSD",
                    days=days
                )
                df = market_data.to_dataframe()
                if not df.empty:
                    logger.info(f"✅ FMP价格数据: {len(df)} 条")
                    return df
            except Exception as e:
                logger.warning(f"FMP获取失败: {e}")
        
        return pd.DataFrame()
    
    async def fetch_macro_data(self, days: int = 365) -> pd.DataFrame:
        """
        获取宏观经济数据
        
        包含: 国债收益率、股票指数、VIX、黄金、美元指数等
        """
        if not self.fmp_api_key:
            logger.warning("⚠️ FMP API密钥未设置，跳过宏观数据")
            return pd.DataFrame()
        
        logger.info("📊 获取宏观经济数据...")
        
        dfs = []
        
        # 1. 国债收益率
        try:
            df_treasury = await self.fmp.get_treasury_rates()
            if not df_treasury.empty:
                dfs.append(df_treasury)
                logger.info(f"  ✓ 国债收益率: {len(df_treasury)} 条")
        except Exception as e:
            logger.warning(f"  ✗ 国债收益率失败: {e}")
        
        # 2. 股票指数
        indices = [
            ("^GSPC", "sp500"),    # S&P 500
            ("^VIX", "vix"),       # VIX恐慌指数
        ]
        
        for symbol, name in indices:
            try:
                df_idx = await self.fmp.get_index_data(symbol=symbol, days=days)
                if not df_idx.empty:
                    dfs.append(df_idx)
                    logger.info(f"  ✓ {name}: {len(df_idx)} 条")
            except Exception as e:
                logger.warning(f"  ✗ {name}失败: {e}")
        
        # 3. 黄金
        try:
            df_gold = await self.fmp.get_commodity_data(symbol="GCUSD", days=days)
            if not df_gold.empty:
                dfs.append(df_gold)
                logger.info(f"  ✓ 黄金: {len(df_gold)} 条")
        except Exception as e:
            logger.warning(f"  ✗ 黄金失败: {e}")
        
        # 4. 美元指数
        try:
            df_dxy = await self.fmp.get_forex_data(symbol="DXY", days=days)
            if not df_dxy.empty:
                dfs.append(df_dxy)
                logger.info(f"  ✓ 美元指数: {len(df_dxy)} 条")
        except Exception as e:
            logger.warning(f"  ✗ 美元指数失败: {e}")
        
        if not dfs:
            return pd.DataFrame()
        
        # 合并所有宏观数据
        df_macro = self._merge_dataframes(dfs)
        logger.info(f"✅ 宏观数据合并: {len(df_macro)} 条, {len(df_macro.columns)} 列")
        
        return df_macro
    
    async def fetch_onchain_data(self, days: int = 90) -> pd.DataFrame:
        """
        获取链上数据
        
        包含: 活跃地址、交易数、哈希率、NVT等
        """
        logger.info("📊 获取链上数据...")
        
        try:
            df_onchain = await self.coinmetrics.get_network_metrics(
                asset="btc",
                start_time=datetime.utcnow() - timedelta(days=days),
                end_time=datetime.utcnow()
            )
            
            if not df_onchain.empty:
                logger.info(f"✅ 链上数据: {len(df_onchain)} 条, {len(df_onchain.columns)} 列")
                return df_onchain
                
        except Exception as e:
            logger.warning(f"链上数据获取失败: {e}")
        
        return pd.DataFrame()
    
    async def fetch_cross_asset(self, days: int = 90) -> pd.DataFrame:
        """
        获取跨市场资产数据
        
        包含: ETH、主要altcoins等
        """
        logger.info("📊 获取跨市场资产数据...")
        
        dfs = []
        
        # 获取ETH价格
        try:
            market_data = await self.coingecko.get_hourly_ohlcv(
                symbol="ethereum",
                days=days
            )
            df_eth = market_data.to_dataframe()
            if not df_eth.empty:
                df_eth.columns = [f"eth_{col}" for col in df_eth.columns]
                dfs.append(df_eth)
                logger.info(f"  ✓ ETH: {len(df_eth)} 条")
        except Exception as e:
            logger.warning(f"  ✗ ETH失败: {e}")
        
        if not dfs:
            return pd.DataFrame()
        
        df_cross = self._merge_dataframes(dfs)
        logger.info(f"✅ 跨市场数据: {len(df_cross)} 条, {len(df_cross.columns)} 列")
        
        return df_cross
    
    # ==================== 主要接口 ====================
    
    async def fetch_all(
        self,
        days: int = 90,
        include_macro: bool = True,
        include_onchain: bool = True,
        include_cross_asset: bool = True,
        resample_to_hourly: bool = True
    ) -> pd.DataFrame:
        """
        获取所有数据并合并
        
        Args:
            days: 历史天数
            include_macro: 是否包含宏观数据
            include_onchain: 是否包含链上数据
            include_cross_asset: 是否包含跨市场数据
            resample_to_hourly: 是否重采样到小时级别
            
        Returns:
            合并后的DataFrame，以timestamp为索引
        """
        logger.info("=" * 50)
        logger.info("开始获取多源数据...")
        logger.info("=" * 50)
        
        # 1. 获取BTC价格数据（核心）
        df_btc = await self.fetch_btc_price(days=days)
        
        if df_btc.empty:
            logger.error("❌ BTC价格数据获取失败，无法继续")
            return pd.DataFrame()
        
        dfs_to_merge = [df_btc]
        
        # 2. 并行获取其他数据
        tasks = []
        
        if include_macro and self.fmp_api_key:
            tasks.append(("macro", self.fetch_macro_data(days=days)))
        
        if include_onchain:
            tasks.append(("onchain", self.fetch_onchain_data(days=days)))
        
        if include_cross_asset:
            tasks.append(("cross_asset", self.fetch_cross_asset(days=days)))
        
        if tasks:
            # 执行所有任务
            results = await asyncio.gather(
                *[task[1] for task in tasks],
                return_exceptions=True
            )
            
            for (name, _), result in zip(tasks, results):
                if isinstance(result, Exception):
                    logger.warning(f"⚠️ {name} 获取失败: {result}")
                elif isinstance(result, pd.DataFrame) and not result.empty:
                    dfs_to_merge.append(result)
        
        # 3. 合并所有数据
        logger.info("\n📊 合并所有数据源...")
        df_merged = self._merge_dataframes(dfs_to_merge, resample_to_hourly=resample_to_hourly)
        
        # 4. 清理数据
        df_merged = self._clean_data(df_merged)
        
        logger.info("=" * 50)
        logger.info(f"✅ 数据获取完成!")
        logger.info(f"   总行数: {len(df_merged)}")
        logger.info(f"   总列数: {len(df_merged.columns)}")
        logger.info(f"   时间范围: {df_merged.index.min()} ~ {df_merged.index.max()}")
        logger.info("=" * 50)
        
        return df_merged
    
    # ==================== 辅助方法 ====================
    
    def _merge_dataframes(
        self,
        dfs: List[pd.DataFrame],
        resample_to_hourly: bool = False
    ) -> pd.DataFrame:
        """
        合并多个DataFrame
        
        Args:
            dfs: DataFrame列表
            resample_to_hourly: 是否重采样到小时级别
            
        Returns:
            合并后的DataFrame
        """
        if not dfs:
            return pd.DataFrame()
        
        if len(dfs) == 1:
            return dfs[0]
        
        # 确保所有索引是datetime类型且无时区
        processed_dfs = []
        for df in dfs:
            if df.empty:
                continue
            
            df = df.copy()
            
            # 确保索引是datetime
            if not isinstance(df.index, pd.DatetimeIndex):
                if "timestamp" in df.columns:
                    df.set_index("timestamp", inplace=True)
                else:
                    continue
            
            # 移除时区
            if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
                df.index = df.index.tz_localize(None)  # type: ignore
            
            # 如果需要重采样日级数据到小时级（前向填充）
            if resample_to_hourly:
                # 检查是否是日级数据
                if len(df) > 1:
                    time_diff = (df.index[1] - df.index[0]).total_seconds()
                    if time_diff >= 86400:  # 日级或更长
                        df = df.resample('h').ffill()
            
            processed_dfs.append(df)
        
        if not processed_dfs:
            return pd.DataFrame()
        
        # 使用outer join合并
        df_merged = processed_dfs[0]
        for df in processed_dfs[1:]:
            df_merged = df_merged.join(df, how='outer')
        
        df_merged.sort_index(inplace=True)
        
        return df_merged
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        清理数据
        
        - 删除全为NaN的行
        - 前向填充缺失值
        - 删除仍有NaN的行
        """
        if df.empty:
            return df
        
        df = df.copy()
        
        # 删除全为NaN的行
        df.dropna(how='all', inplace=True)
        
        # 前向填充
        df.ffill(inplace=True)
        
        # 后向填充剩余的NaN（开头部分）
        df.bfill(inplace=True)
        
        # 删除仍有NaN的行
        df.dropna(inplace=True)
        
        return df
    
    def get_feature_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        获取特征摘要
        
        Args:
            df: 数据DataFrame
            
        Returns:
            特征摘要字典
        """
        if df.empty:
            return {}
        
        # 按类别分组列
        categories = {
            "price": [c for c in df.columns if c in ["open", "high", "low", "close", "volume"]],
            "macro": [c for c in df.columns if any(x in c for x in ["treasury", "sp500", "vix", "dxy", "gc_"])],
            "onchain": [c for c in df.columns if c.startswith("cm_")],
            "cross_asset": [c for c in df.columns if c.startswith("eth_")],
        }
        
        summary = {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "time_range": {
                "start": str(df.index.min()),
                "end": str(df.index.max())
            },
            "categories": {
                name: {
                    "count": len(cols),
                    "columns": cols
                }
                for name, cols in categories.items()
            },
            "missing_values": df.isna().sum().to_dict()
        }
        
        return summary


# ==================== 便捷函数 ====================

async def fetch_training_data(
    fmp_api_key: str = "",
    days: int = 90,
    include_macro: bool = True,
    include_onchain: bool = True
) -> pd.DataFrame:
    """
    便捷函数：获取训练数据
    
    Args:
        fmp_api_key: FMP API密钥
        days: 历史天数
        include_macro: 是否包含宏观数据
        include_onchain: 是否包含链上数据
        
    Returns:
        合并后的训练数据DataFrame
    """
    pipeline = DataPipeline(fmp_api_key=fmp_api_key)
    
    try:
        df = await pipeline.fetch_all(
            days=days,
            include_macro=include_macro,
            include_onchain=include_onchain
        )
        return df
    finally:
        await pipeline.close()
