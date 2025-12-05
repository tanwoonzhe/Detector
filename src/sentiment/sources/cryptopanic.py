"""
CryptoPanic新闻数据源
================================
从CryptoPanic获取加密货币新闻并进行情感分析
"""

import aiohttp
import logging
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any

from .base import SentimentFetcher, SentimentSource, SentimentData
from config import APIConfig

logger = logging.getLogger(__name__)


class CryptoPanicSource(SentimentFetcher):
    """
    CryptoPanic新闻数据源
    
    API: https://cryptopanic.com/api/v1/posts/
    
    免费版限制:
    - 无panic_score
    - 基本过滤功能
    
    返回数据包含:
    - votes: {"liked": N, "disliked": N, "important": N, "lol": N, ...}
    - 可用于计算情感分数
    """
    
    def __init__(self, api_key: str = ""):
        super().__init__(SentimentSource.CRYPTOPANIC)
        self.api_key = api_key or APIConfig.CRYPTOPANIC_API_KEY
        self.base_url = APIConfig.CRYPTOPANIC_BASE_URL
        self._session: Optional[aiohttp.ClientSession] = None
    
    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session
    
    def _calculate_sentiment_from_votes(self, votes: Dict[str, int]) -> float:
        """
        从投票数据计算情感分数
        
        正面: liked, important, saved
        负面: disliked, toxic
        中性: lol, comments
        """
        positive = votes.get("liked", 0) + votes.get("important", 0) * 2 + votes.get("saved", 0)
        negative = votes.get("disliked", 0) + votes.get("toxic", 0) * 2
        
        total = positive + negative
        if total == 0:
            return 0.0
        
        # 计算情感分数 (-1 到 +1)
        score = (positive - negative) / total
        return max(-1.0, min(1.0, score))
    
    def _calculate_confidence(self, votes: Dict[str, int]) -> float:
        """
        根据投票总数计算置信度
        投票越多，置信度越高
        """
        total = sum(votes.values())
        # 使用对数缩放，10票以上置信度开始显著
        import math
        return min(1.0, math.log10(total + 1) / 2)
    
    async def fetch_sentiment(
        self, 
        symbol: str = "BTC",
        hours_back: int = 24
    ) -> List[SentimentData]:
        """获取CryptoPanic新闻数据"""
        if not self.api_key:
            logger.warning("未配置CryptoPanic API密钥，使用公开端点")
        
        session = await self._get_session()
        
        params = {
            "auth_token": self.api_key,
            "currencies": symbol,
            "kind": "news",
            "public": "true"
        }
        
        try:
            async with session.get(
                f"{self.base_url}/posts/",
                params=params
            ) as response:
                if response.status == 401:
                    logger.warning("CryptoPanic API密钥无效或未提供")
                    return []
                response.raise_for_status()
                data = await response.json()
        except aiohttp.ClientError as e:
            logger.error(f"CryptoPanic API请求失败: {e}")
            return []
        
        sentiment_list = []
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        
        for item in data.get("results", []):
            # 解析时间
            published_at = item.get("published_at", "")
            try:
                timestamp = datetime.fromisoformat(published_at.replace("Z", "+00:00"))
                timestamp = timestamp.replace(tzinfo=None)  # 移除时区信息
            except:
                continue
            
            # 过滤时间范围
            if timestamp < cutoff_time:
                continue
            
            votes = item.get("votes", {})
            
            sentiment_list.append(SentimentData(
                timestamp=timestamp,
                score=self._calculate_sentiment_from_votes(votes),
                source=self.source.value,
                confidence=self._calculate_confidence(votes),
                raw_text=item.get("title", ""),
                metadata={
                    "url": item.get("url"),
                    "domain": item.get("domain"),
                    "votes": votes,
                    "kind": item.get("kind")
                }
            ))
        
        logger.info(f"获取到 {len(sentiment_list)} 条CryptoPanic新闻")
        return sentiment_list
    
    async def get_current_sentiment(self, symbol: str = "BTC") -> Optional[SentimentData]:
        """获取最新新闻情感"""
        sentiments = await self.fetch_sentiment(symbol, hours_back=1)
        if not sentiments:
            return None
        
        # 返回最新的
        return max(sentiments, key=lambda x: x.timestamp)
    
    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()
