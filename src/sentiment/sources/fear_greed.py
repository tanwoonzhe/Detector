"""
Fear & Greed Index数据源
================================
从Alternative.me获取加密货币恐惧贪婪指数
"""

import aiohttp
import logging
from datetime import datetime, timedelta
from typing import List, Optional

from .base import SentimentFetcher, SentimentSource, SentimentData
from config import APIConfig

logger = logging.getLogger(__name__)


class FearGreedSource(SentimentFetcher):
    """
    恐惧贪婪指数数据源
    
    指数范围: 0-100
    - 0-24: 极度恐惧 (Extreme Fear)
    - 25-49: 恐惧 (Fear)
    - 50: 中性 (Neutral)
    - 51-74: 贪婪 (Greed)
    - 75-100: 极度贪婪 (Extreme Greed)
    
    API: https://api.alternative.me/fng/
    限制: 免费无限制
    """
    
    def __init__(self):
        super().__init__(SentimentSource.FEAR_GREED)
        self.api_url = APIConfig.FEAR_GREED_URL
        self._session: Optional[aiohttp.ClientSession] = None
    
    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session
    
    def _normalize_score(self, value: int) -> float:
        """
        将0-100的恐惧贪婪指数转换为-1到+1的标准化分数
        
        0 -> -1 (极度恐惧)
        50 -> 0 (中性)
        100 -> +1 (极度贪婪)
        """
        return (value - 50) / 50
    
    def _get_classification(self, value: int) -> str:
        """获取情感分类"""
        if value <= 24:
            return "Extreme Fear"
        elif value <= 49:
            return "Fear"
        elif value == 50:
            return "Neutral"
        elif value <= 74:
            return "Greed"
        else:
            return "Extreme Greed"
    
    async def fetch_sentiment(
        self, 
        symbol: str = "BTC",
        hours_back: int = 24
    ) -> List[SentimentData]:
        """
        获取历史恐惧贪婪指数
        
        注意: Fear & Greed Index是日级别数据
        """
        days = max(1, hours_back // 24)
        
        session = await self._get_session()
        
        try:
            async with session.get(
                self.api_url,
                params={"limit": days, "format": "json"}
            ) as response:
                response.raise_for_status()
                data = await response.json()
        except aiohttp.ClientError as e:
            logger.error(f"Fear & Greed API请求失败: {e}")
            return []
        
        sentiment_list = []
        
        for item in data.get("data", []):
            value = int(item.get("value", 50))
            timestamp = datetime.fromtimestamp(int(item.get("timestamp", 0)))
            
            sentiment_list.append(SentimentData(
                timestamp=timestamp,
                score=self._normalize_score(value),
                source=self.source.value,
                confidence=1.0,  # 官方指数，置信度为1
                raw_text=None,
                metadata={
                    "raw_value": value,
                    "classification": item.get("value_classification", self._get_classification(value)),
                    "time_until_update": item.get("time_until_update")
                }
            ))
        
        logger.info(f"获取到 {len(sentiment_list)} 条Fear & Greed数据")
        return sentiment_list
    
    async def get_current_sentiment(self, symbol: str = "BTC") -> Optional[SentimentData]:
        """获取当前恐惧贪婪指数"""
        session = await self._get_session()
        
        try:
            async with session.get(self.api_url) as response:
                response.raise_for_status()
                data = await response.json()
        except aiohttp.ClientError as e:
            logger.error(f"Fear & Greed API请求失败: {e}")
            return None
        
        if not data.get("data"):
            return None
        
        current = data["data"][0]
        value = int(current.get("value", 50))
        
        return SentimentData(
            timestamp=datetime.now(),
            score=self._normalize_score(value),
            source=self.source.value,
            confidence=1.0,
            metadata={
                "raw_value": value,
                "classification": current.get("value_classification", self._get_classification(value))
            }
        )
    
    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()
