"""
SQLite缓存管理器
================================
用于缓存API数据，减少请求次数，处理速率限制
"""

import sqlite3
import json
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from pathlib import Path
import logging

from config import DATA_DIR

logger = logging.getLogger(__name__)


class CacheManager:
    """
    SQLite缓存管理器
    缓存OHLCV数据和情感数据，支持过期策略
    """
    
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or DATA_DIR / "btcusdt_cache.db"
        self._init_database()
    
    def _init_database(self):
        """初始化数据库表"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # OHLCV数据缓存表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ohlcv_cache (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    source TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    volume REAL NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(symbol, source, timestamp)
                )
            """)
            
            # 情感数据缓存表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sentiment_cache (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    score REAL NOT NULL,
                    raw_data TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(source, timestamp)
                )
            """)
            
            # API请求记录表（用于速率限制）
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS api_requests (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source TEXT NOT NULL,
                    endpoint TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    success INTEGER DEFAULT 1
                )
            """)
            
            # 创建索引
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_ohlcv_symbol_source 
                ON ohlcv_cache(symbol, source, timestamp)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_sentiment_source 
                ON sentiment_cache(source, timestamp)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_api_requests 
                ON api_requests(source, timestamp)
            """)
            
            conn.commit()
            logger.info(f"数据库初始化完成: {self.db_path}")
    
    def save_ohlcv(
        self, 
        symbol: str, 
        source: str, 
        df: pd.DataFrame
    ):
        """
        保存OHLCV数据到缓存
        
        Args:
            symbol: 交易对符号
            source: 数据源
            df: 包含OHLCV数据的DataFrame
        """
        if df.empty:
            return
        
        with sqlite3.connect(self.db_path) as conn:
            for timestamp, row in df.iterrows():
                try:
                    conn.execute("""
                        INSERT OR REPLACE INTO ohlcv_cache 
                        (symbol, source, timestamp, open, high, low, close, volume)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        symbol,
                        source,
                        timestamp.isoformat() if isinstance(timestamp, datetime) else str(timestamp),
                        row['open'],
                        row['high'],
                        row['low'],
                        row['close'],
                        row['volume']
                    ))
                except Exception as e:
                    logger.warning(f"保存OHLCV数据失败: {e}")
            
            conn.commit()
            logger.info(f"保存 {len(df)} 条OHLCV数据到缓存")
    
    def get_ohlcv(
        self, 
        symbol: str, 
        source: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        从缓存获取OHLCV数据
        
        Args:
            symbol: 交易对符号
            source: 数据源
            start_time: 开始时间
            end_time: 结束时间
            
        Returns:
            DataFrame
        """
        query = """
            SELECT timestamp, open, high, low, close, volume
            FROM ohlcv_cache
            WHERE symbol = ? AND source = ?
        """
        params = [symbol, source]
        
        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time.isoformat())
        
        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time.isoformat())
        
        query += " ORDER BY timestamp ASC"
        
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(query, conn, params=tuple(params))
        
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
        
        return df
    
    def save_sentiment(
        self, 
        source: str, 
        timestamp: datetime, 
        score: float,
        raw_data: Optional[Dict] = None
    ):
        """保存情感数据"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO sentiment_cache 
                (source, timestamp, score, raw_data)
                VALUES (?, ?, ?, ?)
            """, (
                source,
                timestamp.isoformat(),
                score,
                json.dumps(raw_data) if raw_data else None
            ))
            conn.commit()
    
    def get_sentiment(
        self, 
        source: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> pd.DataFrame:
        """获取情感数据"""
        query = """
            SELECT timestamp, score, raw_data
            FROM sentiment_cache
            WHERE source = ?
        """
        params = [source]
        
        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time.isoformat())
        
        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time.isoformat())
        
        query += " ORDER BY timestamp ASC"
        
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(query, conn, params=tuple(params))
        
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
        
        return df
    
    def record_api_request(self, source: str, endpoint: str, success: bool = True):
        """记录API请求（用于速率限制跟踪）"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO api_requests (source, endpoint, success)
                VALUES (?, ?, ?)
            """, (source, endpoint, 1 if success else 0))
            conn.commit()
    
    def get_recent_request_count(self, source: str, minutes: int = 1) -> int:
        """获取最近N分钟内的请求数"""
        cutoff = datetime.now() - timedelta(minutes=minutes)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT COUNT(*) FROM api_requests
                WHERE source = ? AND timestamp >= ?
            """, (source, cutoff.isoformat()))
            return cursor.fetchone()[0]
    
    def clear_old_data(self, days: int = 180):
        """清理旧数据"""
        cutoff = datetime.now() - timedelta(days=days)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                DELETE FROM ohlcv_cache WHERE created_at < ?
            """, (cutoff.isoformat(),))
            conn.execute("""
                DELETE FROM sentiment_cache WHERE created_at < ?
            """, (cutoff.isoformat(),))
            conn.execute("""
                DELETE FROM api_requests WHERE timestamp < ?
            """, (cutoff.isoformat(),))
            conn.commit()
            logger.info(f"已清理 {days} 天前的缓存数据")


# 全局缓存实例
cache_manager = CacheManager()
