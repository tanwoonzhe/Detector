"""
Reddit数据源
================================
从Reddit获取加密货币社区讨论情感
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Optional
import aiohttp

from .base import SentimentFetcher, SentimentSource, SentimentData
from config import APIConfig

logger = logging.getLogger(__name__)


class RedditSource(SentimentFetcher):
    """
    Reddit数据源
    
    获取以下subreddit的帖子:
    - r/Bitcoin
    - r/CryptoCurrency
    - r/BitcoinMarkets
    
    使用Reddit JSON API (无需OAuth)
    """
    
    SUBREDDITS = ["Bitcoin", "CryptoCurrency", "BitcoinMarkets"]
    
    def __init__(self):
        super().__init__(SentimentSource.REDDIT)
        self._session: Optional[aiohttp.ClientSession] = None
    
    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                headers={"User-Agent": APIConfig.REDDIT_USER_AGENT}
            )
        return self._session
    
    def _calculate_sentiment_from_score(self, score: int, upvote_ratio: float) -> float:
        """
        从帖子分数和点赞比例计算情感
        
        假设:
        - 高分+高点赞比 = 社区认可 (可能是看涨信号)
        - 低分或低点赞比 = 争议或负面
        """
        # 点赞比例转换: 0.5 -> -1, 1.0 -> +1
        ratio_score = (upvote_ratio - 0.5) * 2
        
        # 分数影响置信度而非方向
        return ratio_score
    
    def _calculate_confidence(self, score: int, num_comments: int) -> float:
        """
        根据互动量计算置信度
        """
        import math
        engagement = score + num_comments
        return min(1.0, math.log10(engagement + 1) / 3)
    
    async def _fetch_subreddit(
        self, 
        subreddit: str, 
        limit: int = 25,
        sort: str = "hot"
    ) -> List[dict]:
        """获取单个subreddit的帖子"""
        session = await self._get_session()
        url = f"https://www.reddit.com/r/{subreddit}/{sort}.json"
        
        try:
            async with session.get(
                url,
                params={"limit": limit, "raw_json": 1}
            ) as response:
                if response.status == 429:
                    logger.warning(f"Reddit速率限制，等待...")
                    await asyncio.sleep(60)
                    return []
                response.raise_for_status()
                data = await response.json()
                return data.get("data", {}).get("children", [])
        except aiohttp.ClientError as e:
            logger.error(f"Reddit API请求失败 ({subreddit}): {e}")
            return []
    
    async def fetch_sentiment(
        self, 
        symbol: str = "BTC",
        hours_back: int = 24
    ) -> List[SentimentData]:
        """获取Reddit情感数据"""
        sentiment_list = []
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        
        # 并发获取所有subreddit
        tasks = [
            self._fetch_subreddit(sub, limit=50)
            for sub in self.SUBREDDITS
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for subreddit, result in zip(self.SUBREDDITS, results):
            if isinstance(result, Exception):
                logger.error(f"获取 r/{subreddit} 失败: {result}")
                continue
            
            if isinstance(result, list):
                for post in result:
                    post_data = post.get("data", {})
                    
                    # 解析时间
                    created_utc = post_data.get("created_utc", 0)
                    timestamp = datetime.fromtimestamp(created_utc)
                    
                    if timestamp < cutoff_time:
                        continue
                    
                    # 过滤与BTC相关的帖子
                    title = post_data.get("title", "").lower()
                    if symbol.lower() not in title and "bitcoin" not in title and "btc" not in title:
                        if subreddit != "Bitcoin":  # r/Bitcoin全部算BTC相关
                            continue
                    
                    score = post_data.get("score", 0)
                    upvote_ratio = post_data.get("upvote_ratio", 0.5)
                    num_comments = post_data.get("num_comments", 0)
                    
                    sentiment_list.append(SentimentData(
                        timestamp=timestamp,
                        score=self._calculate_sentiment_from_score(score, upvote_ratio),
                        source=self.source.value,
                        confidence=self._calculate_confidence(score, num_comments),
                        raw_text=post_data.get("title", ""),
                        metadata={
                            "subreddit": subreddit,
                            "score": score,
                            "upvote_ratio": upvote_ratio,
                            "num_comments": num_comments,
                            "url": f"https://reddit.com{post_data.get('permalink', '')}"
                        }
                    ))
        
        logger.info(f"获取到 {len(sentiment_list)} 条Reddit帖子")
        return sentiment_list
    
    async def get_current_sentiment(self, symbol: str = "BTC") -> Optional[SentimentData]:
        """获取最新Reddit情感"""
        sentiments = await self.fetch_sentiment(symbol, hours_back=2)
        if not sentiments:
            return None
        
        # 计算加权平均情感
        total_score = 0
        total_weight = 0
        
        for s in sentiments:
            weight = s.confidence
            total_score += s.score * weight
            total_weight += weight
        
        if total_weight == 0:
            return None
        
        avg_score = total_score / total_weight
        
        return SentimentData(
            timestamp=datetime.now(),
            score=avg_score,
            source=self.source.value,
            confidence=min(1.0, len(sentiments) / 20),  # 帖子越多置信度越高
            metadata={"post_count": len(sentiments)}
        )
    
    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()
