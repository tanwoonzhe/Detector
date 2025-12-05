"""
数据采集抽象基类
================================
定义统一的数据获取接口，支持多数据源切换（CoinGecko/Binance）
使用工厂模式创建具体实现
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Any
from enum import Enum
import pandas as pd


class DataSource(Enum):
    """数据源枚举"""
    COINGECKO = "coingecko"
    BINANCE = "binance"


@dataclass
class OHLCV:
    """OHLCV数据结构"""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume
        }


@dataclass
class MarketData:
    """市场数据结构（包含额外信息）"""
    symbol: str
    ohlcv_data: List[OHLCV]
    market_cap: Optional[float] = None
    total_volume_24h: Optional[float] = None
    price_change_24h: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dataframe(self) -> pd.DataFrame:
        """转换为DataFrame"""
        if not self.ohlcv_data:
            return pd.DataFrame()
        
        data = [ohlcv.to_dict() for ohlcv in self.ohlcv_data]
        df = pd.DataFrame(data)
        df.set_index("timestamp", inplace=True)
        df.sort_index(inplace=True)
        return df


@dataclass
class RateLimitInfo:
    """速率限制信息"""
    calls_per_minute: int
    daily_limit: Optional[int] = None
    remaining_calls: Optional[int] = None
    reset_time: Optional[datetime] = None


class DataFetcher(ABC):
    """
    数据获取抽象基类
    所有数据源必须实现此接口
    """
    
    def __init__(self, source: DataSource):
        self.source = source
        self._last_request_time: Optional[datetime] = None
    
    @abstractmethod
    async def get_hourly_ohlcv(
        self, 
        symbol: str, 
        days: int = 90,
        vs_currency: str = "usd"
    ) -> MarketData:
        """
        获取小时级OHLCV数据
        
        Args:
            symbol: 交易对符号
            days: 历史天数（2-90天返回小时数据）
            vs_currency: 计价货币
            
        Returns:
            MarketData对象
        """
        pass
    
    @abstractmethod
    async def get_current_price(self, symbol: str) -> float:
        """获取当前价格"""
        pass
    
    @abstractmethod
    def get_rate_limit(self) -> RateLimitInfo:
        """获取速率限制信息"""
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """检查API连接状态"""
        pass
    
    def get_source_name(self) -> str:
        """获取数据源名称"""
        return self.source.value


class DataFetcherFactory:
    """
    数据获取器工厂类
    用于创建和管理不同数据源的实例
    """
    
    _fetchers: Dict[DataSource, type] = {}
    _instances: Dict[DataSource, DataFetcher] = {}
    
    @classmethod
    def register(cls, source: DataSource, fetcher_class: type):
        """注册数据源"""
        cls._fetchers[source] = fetcher_class
    
    @classmethod
    def create(cls, source: DataSource, **kwargs) -> DataFetcher:
        """
        创建数据获取器实例
        
        Args:
            source: 数据源类型
            **kwargs: 传递给构造函数的参数
            
        Returns:
            DataFetcher实例
        """
        if source not in cls._fetchers:
            raise ValueError(f"未注册的数据源: {source.value}")
        
        # 单例模式（可选）
        if source not in cls._instances:
            cls._instances[source] = cls._fetchers[source](**kwargs)
        
        return cls._instances[source]
    
    @classmethod
    def get_available_sources(cls) -> List[str]:
        """获取所有可用数据源"""
        return [source.value for source in cls._fetchers.keys()]
