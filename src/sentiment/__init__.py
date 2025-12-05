"""情感分析模块"""
from .sources import (
    SentimentSource,
    SentimentData,
    SentimentFetcher,
    FearGreedSource,
    CryptoPanicSource,
    RedditSource
)
from .analyzer import SentimentAnalyzer, AnalysisResult, sentiment_analyzer
from .aggregator import SentimentAggregator, sentiment_aggregator

__all__ = [
    # 数据源
    "SentimentSource",
    "SentimentData",
    "SentimentFetcher",
    "FearGreedSource",
    "CryptoPanicSource",
    "RedditSource",
    # 分析器
    "SentimentAnalyzer",
    "AnalysisResult",
    "sentiment_analyzer",
    # 聚合器
    "SentimentAggregator",
    "sentiment_aggregator"
]
