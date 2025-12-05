"""
情感数据源基类
================================
定义情感数据获取的统一接口
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Dict, Any
from enum import Enum


class SentimentSource(Enum):
    """情感数据源枚举"""
    FEAR_GREED = "fear_greed"
    CRYPTOPANIC = "cryptopanic"
    REDDIT = "reddit"
    NEWS = "news"


@dataclass
class SentimentData:
    """情感数据结构"""
    timestamp: datetime
    score: float  # -1 (极度悲观) 到 +1 (极度乐观)
    source: str
    confidence: float = 1.0
    raw_text: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "score": self.score,
            "source": self.source,
            "confidence": self.confidence,
            "raw_text": self.raw_text,
            "metadata": self.metadata
        }


class SentimentFetcher(ABC):
    """
    情感数据获取抽象基类
    """
    
    def __init__(self, source: SentimentSource):
        self.source = source
    
    @abstractmethod
    async def fetch_sentiment(
        self, 
        symbol: str = "BTC",
        hours_back: int = 24
    ) -> List[SentimentData]:
        """
        获取情感数据
        
        Args:
            symbol: 币种符号
            hours_back: 回溯小时数
            
        Returns:
            情感数据列表
        """
        pass
    
    @abstractmethod
    async def get_current_sentiment(self, symbol: str = "BTC") -> Optional[SentimentData]:
        """获取当前情感"""
        pass
    
    def get_source_name(self) -> str:
        return self.source.value
