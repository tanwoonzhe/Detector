"""
情感数据聚合器
================================
整合多个情感数据源，按时间对齐，生成统一的情感特征
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import pandas as pd
import numpy as np

from .sources import (
    SentimentData,
    FearGreedSource,
    CryptoPanicSource,
    RedditSource
)
from .analyzer import sentiment_analyzer, AnalysisResult
from src.data_collection import cache_manager
from config import SentimentConfig

logger = logging.getLogger(__name__)


class SentimentAggregator:
    """
    情感数据聚合器
    
    功能:
    1. 从多个数据源收集情感数据
    2. 使用NLP模型分析原始文本
    3. 按时间对齐到小时级别
    4. 加权聚合生成统一情感分数
    """
    
    def __init__(self):
        # 初始化数据源
        self.fear_greed = FearGreedSource()
        self.cryptopanic = CryptoPanicSource()
        self.reddit = RedditSource()
        
        # 权重配置
        self.weights = SentimentConfig.SENTIMENT_WEIGHTS
    
    async def fetch_all_sources(
        self, 
        symbol: str = "BTC",
        hours_back: int = 24
    ) -> Dict[str, List[SentimentData]]:
        """
        并发获取所有数据源
        
        Returns:
            {source_name: [SentimentData, ...]}
        """
        logger.info(f"获取 {symbol} 的情感数据 (过去 {hours_back} 小时)...")
        
        # 并发获取
        tasks = [
            self.fear_greed.fetch_sentiment(symbol, hours_back),
            self.cryptopanic.fetch_sentiment(symbol, hours_back),
            self.reddit.fetch_sentiment(symbol, hours_back)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        data = {}
        source_names = ["fear_greed", "news", "reddit"]
        
        for name, result in zip(source_names, results):
            if isinstance(result, BaseException):
                logger.error(f"获取 {name} 数据失败: {result}")
                data[name] = []
            else:
                data[name] = result
                logger.info(f"  - {name}: {len(data[name])} 条数据")
        
        return data
    
    def _analyze_texts(
        self, 
        sentiment_data: List[SentimentData],
        use_cryptobert: bool = True
    ) -> List[SentimentData]:
        """
        对原始文本进行NLP情感分析
        """
        texts_to_analyze = []
        indices = []
        
        for i, sd in enumerate(sentiment_data):
            if sd.raw_text:
                texts_to_analyze.append(sd.raw_text)
                indices.append(i)
        
        if not texts_to_analyze:
            return sentiment_data
        
        # 批量分析
        results = sentiment_analyzer.analyze_batch(
            texts_to_analyze, 
            use_cryptobert=use_cryptobert
        )
        
        # 更新情感分数
        for idx, result in zip(indices, results):
            # 混合原始分数和NLP分数
            original_score = sentiment_data[idx].score
            nlp_score = result.score
            nlp_confidence = result.confidence
            
            # 加权平均: NLP权重更高
            sentiment_data[idx].score = (
                original_score * 0.3 + nlp_score * 0.7
            )
            sentiment_data[idx].confidence = (
                sentiment_data[idx].confidence * 0.5 + nlp_confidence * 0.5
            )
        
        return sentiment_data
    
    def _align_to_hourly(
        self, 
        sentiment_data: List[SentimentData]
    ) -> pd.DataFrame:
        """
        将情感数据对齐到小时级别
        
        Returns:
            DataFrame with hourly sentiment scores
        """
        if not sentiment_data:
            return pd.DataFrame()
        
        # 转换为DataFrame
        records = [sd.to_dict() for sd in sentiment_data]
        df = pd.DataFrame(records)
        
        if df.empty:
            return df
        
        # 对齐到小时
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['timestamp'].apply(lambda x: x.floor('H') if hasattr(x, 'floor') else x)
        
        # 按小时分组，加权平均
        hourly = df.groupby('hour').apply(
            lambda x: pd.Series({
                'score': np.average(x['score'], weights=x['confidence']) if x['confidence'].sum() > 0 else 0,
                'confidence': x['confidence'].mean(),
                'count': len(x)
            })
        )
        
        hourly.index.name = 'timestamp'
        return hourly.reset_index()
    
    async def get_hourly_sentiment(
        self, 
        symbol: str = "BTC",
        hours_back: int = 168,  # 7天
        analyze_texts: bool = True
    ) -> pd.DataFrame:
        """
        获取小时级情感数据
        
        Args:
            symbol: 币种符号
            hours_back: 回溯小时数
            analyze_texts: 是否对文本进行NLP分析
            
        Returns:
            DataFrame columns: [timestamp, fear_greed, news, reddit, composite]
        """
        # 获取所有数据源
        all_data = await self.fetch_all_sources(symbol, hours_back)
        
        # 对文本进行NLP分析
        if analyze_texts:
            logger.info("执行NLP情感分析...")
            all_data["news"] = self._analyze_texts(
                all_data["news"], 
                use_cryptobert=True
            )
            all_data["reddit"] = self._analyze_texts(
                all_data["reddit"], 
                use_cryptobert=False  # Reddit用VADER，更适合非正式语言
            )
        
        # 对齐到小时
        hourly_data = {}
        for source, data in all_data.items():
            hourly_df = self._align_to_hourly(data)
            if not hourly_df.empty:
                hourly_df = hourly_df.set_index('timestamp')
                hourly_data[source] = hourly_df['score']
        
        if not hourly_data:
            logger.warning("没有获取到任何情感数据")
            return pd.DataFrame()
        
        # 合并所有数据源
        result = pd.DataFrame(hourly_data)
        
        # 填充缺失值
        # Fear & Greed是日级别，向前填充
        if 'fear_greed' in result.columns:
            result['fear_greed'] = result['fear_greed'].ffill()
        
        # 其他源用0填充
        result = result.fillna(0)
        
        # 计算复合情感分数
        result['composite'] = 0
        for source, weight in self.weights.items():
            if source in result.columns:
                result['composite'] += result[source] * weight
        
        # 归一化到 [-1, 1]
        result['composite'] = result['composite'].clip(-1, 1)
        
        result = result.reset_index()
        result.rename(columns={'index': 'timestamp'}, inplace=True)
        
        logger.info(f"生成 {len(result)} 条小时级情感数据")
        
        # 缓存结果
        for _, row in result.iterrows():
            cache_manager.save_sentiment(
                "composite",
                row['timestamp'],
                row['composite'],
                {"sources": {s: row.get(s, 0) for s in self.weights.keys()}}
            )
        
        return result
    
    async def get_current_composite_sentiment(
        self, 
        symbol: str = "BTC"
    ) -> Tuple[float, Dict[str, float]]:
        """
        获取当前复合情感分数
        
        Returns:
            (composite_score, {source: score})
        """
        # 获取各源当前情感
        tasks = [
            self.fear_greed.get_current_sentiment(symbol),
            self.cryptopanic.get_current_sentiment(symbol),
            self.reddit.get_current_sentiment(symbol)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        scores = {}
        source_names = ["fear_greed", "news", "reddit"]
        
        for name, result in zip(source_names, results):
            if isinstance(result, BaseException) or result is None:
                scores[name] = 0.0
            elif hasattr(result, 'score'):
                scores[name] = result.score
            else:
                scores[name] = 0.0
        
        # 计算复合分数
        composite = sum(
            scores.get(source, 0) * weight
            for source, weight in self.weights.items()
        )
        composite = max(-1, min(1, composite))
        
        return composite, scores
    
    async def close(self):
        """关闭所有数据源连接"""
        await asyncio.gather(
            self.fear_greed.close(),
            self.cryptopanic.close(),
            self.reddit.close()
        )


# 全局聚合器实例
sentiment_aggregator = SentimentAggregator()
