"""数据采集模块 - Multi-source Data Collection"""
from .base import (
    DataSource,
    OHLCV,
    MarketData,
    RateLimitInfo,
    DataFetcher,
    DataFetcherFactory
)
from .cache import CacheManager, cache_manager
from .fmp_fetcher import FMPFetcher
from .coinmetrics_fetcher import CoinMetricsFetcher
from .data_pipeline import DataPipeline, fetch_training_data
from .binance_historical import BinanceHistoricalFetcher, download_btc_historical, load_btc_historical
from .fred_fetcher import FREDFetcher
from .kaggle_fetcher import KaggleFetcher, load_kaggle_btc
from .hf_loader_multi import load_hf_btc_multi_granularity

__all__ = [
    # Base classes
    "DataSource",
    "OHLCV",
    "MarketData",
    "RateLimitInfo",
    "DataFetcher",
    "DataFetcherFactory",
    # Cache
    "CacheManager",
    "cache_manager",
    # Data Fetchers
    "FMPFetcher",
    "CoinMetricsFetcher",
    "BinanceHistoricalFetcher",
    "FREDFetcher",
    "KaggleFetcher",
    # Convenience functions
    "download_btc_historical",
    "load_btc_historical",
    "load_kaggle_btc",
    "load_hf_btc_multi_granularity",
    # Pipeline
    "DataPipeline",
    "fetch_training_data"
]
