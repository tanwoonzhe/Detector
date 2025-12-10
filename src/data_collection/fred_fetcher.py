"""
FRED (Federal Reserve Economic Data) 宏观数据获取器
================================
从美联储经济数据库获取宏观经济指标

数据源: https://fred.stlouisfed.org/
免费API: 需要注册获取API Key

支持指标:
- 利率: 联邦基金利率, 国债收益率
- 通胀: CPI, PCE
- 货币: M2供应量
- 市场: VIX (通过FRED镜像)
- 就业: 失业率
"""

import asyncio
import aiohttp
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any
import pandas as pd

logger = logging.getLogger(__name__)

# FRED API 基础URL
FRED_BASE_URL = "https://api.stlouisfed.org/fred"

# 常用宏观指标系列
MACRO_SERIES = {
    # 利率
    "DFF": "联邦基金有效利率",
    "DGS10": "10年期国债收益率",
    "DGS2": "2年期国债收益率",
    "DGS30": "30年期国债收益率",
    "T10Y2Y": "10年-2年期限利差",
    "T10Y3M": "10年-3月期限利差",
    
    # 通胀
    "CPIAUCSL": "CPI-所有城市消费者",
    "CPILFESL": "核心CPI (不含食品能源)",
    "PCEPI": "PCE价格指数",
    "PCEPILFE": "核心PCE (不含食品能源)",
    
    # 货币供应
    "M2SL": "M2货币供应量",
    "M1SL": "M1货币供应量",
    
    # 市场指标
    "VIXCLS": "VIX恐慌指数",
    "SP500": "标普500指数",
    "NASDAQCOM": "纳斯达克综合指数",
    "DTWEXBGS": "美元指数 (广义)",
    
    # 就业
    "UNRATE": "失业率",
    "PAYEMS": "非农就业人数",
    "ICSA": "首次申请失业救济人数",
    
    # 经济活动
    "INDPRO": "工业生产指数",
    "RSAFS": "零售销售",
    "DGORDER": "耐用品订单",
    
    # 房地产
    "CSUSHPISA": "Case-Shiller房价指数",
    
    # 大宗商品
    "DCOILWTICO": "WTI原油价格",
    "GOLDAMGBD228NLBM": "黄金价格(伦敦)",
}

# 与BTC相关性较高的核心指标
BTC_RELATED_SERIES = [
    "DFF",       # 联邦基金利率 - 货币政策
    "DGS10",     # 10年期国债 - 无风险利率
    "DGS2",      # 2年期国债
    "T10Y2Y",    # 期限利差 - 经济预期
    "CPIAUCSL",  # CPI - 通胀
    "M2SL",      # M2 - 流动性
    "VIXCLS",    # VIX - 风险情绪
    "DTWEXBGS",  # 美元指数
    "DCOILWTICO", # 原油 - 通胀预期
]


class FREDFetcher:
    """
    FRED 宏观数据获取器
    
    需要FRED API Key，可在以下网址免费注册:
    https://fred.stlouisfed.org/docs/api/api_key.html
    """
    
    def __init__(self, api_key: str = ""):
        """
        初始化
        
        Args:
            api_key: FRED API Key
        """
        self.api_key = api_key
        self.base_url = FRED_BASE_URL
        self._session: Optional[aiohttp.ClientSession] = None
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """获取HTTP会话"""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session
    
    async def close(self):
        """关闭会话"""
        if self._session and not self._session.closed:
            await self._session.close()
    
    async def _request(self, endpoint: str, params: Optional[Dict] = None) -> Dict:
        """发送API请求"""
        if not self.api_key:
            raise ValueError("FRED API Key未设置，请在.env中设置FRED_API_KEY")
        
        session = await self._get_session()
        url = f"{self.base_url}/{endpoint}"
        
        if params is None:
            params = {}
        params["api_key"] = self.api_key
        params["file_type"] = "json"
        
        try:
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    text = await response.text()
                    logger.error(f"FRED API错误 {response.status}: {text}")
                    return {}
        except Exception as e:
            logger.error(f"FRED API请求失败: {e}")
            return {}
    
    async def get_series(
        self,
        series_id: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        frequency: Optional[str] = None
    ) -> pd.DataFrame:
        """
        获取单个经济指标时间序列
        
        Args:
            series_id: FRED系列ID (如 "DFF", "DGS10")
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
            frequency: 频率 (d=日, w=周, m=月, q=季, a=年)
            
        Returns:
            DataFrame with date index and value column
        """
        if start_date is None:
            start_date = "2010-01-01"
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
        
        params = {
            "series_id": series_id,
            "observation_start": start_date,
            "observation_end": end_date,
        }
        
        if frequency:
            params["frequency"] = frequency
        
        logger.info(f"📥 获取FRED数据: {series_id} ({MACRO_SERIES.get(series_id, '')})")
        
        data = await self._request("series/observations", params)
        
        if not data or "observations" not in data:
            logger.warning(f"FRED返回空数据: {series_id}")
            return pd.DataFrame()
        
        # 解析数据
        records = []
        for obs in data["observations"]:
            try:
                value = float(obs["value"]) if obs["value"] != "." else None
                records.append({
                    "timestamp": pd.to_datetime(obs["date"]),
                    f"fred_{series_id.lower()}": value
                })
            except (ValueError, KeyError):
                continue
        
        if not records:
            return pd.DataFrame()
        
        df = pd.DataFrame(records)
        df.set_index("timestamp", inplace=True)
        df.sort_index(inplace=True)
        
        # 删除NaN
        df.dropna(inplace=True)
        
        logger.info(f"✅ {series_id}: {len(df)} 条记录 ({df.index.min().date()} ~ {df.index.max().date()})")
        
        return df
    
    async def get_multiple_series(
        self,
        series_ids: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        获取多个经济指标并合并
        
        Args:
            series_ids: 系列ID列表，默认使用BTC相关指标
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            合并后的DataFrame
        """
        if series_ids is None:
            series_ids = BTC_RELATED_SERIES
        
        logger.info(f"📥 批量获取FRED数据: {len(series_ids)} 个指标")
        
        # 并行获取所有数据
        tasks = [
            self.get_series(sid, start_date, end_date)
            for sid in series_ids
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 合并数据
        dfs = []
        for sid, result in zip(series_ids, results):
            if isinstance(result, Exception):
                logger.warning(f"获取 {sid} 失败: {result}")
            elif isinstance(result, pd.DataFrame) and not result.empty:
                dfs.append(result)
        
        if not dfs:
            logger.warning("没有获取到任何FRED数据")
            return pd.DataFrame()
        
        # 合并所有数据
        df_merged = dfs[0]
        for df in dfs[1:]:
            df_merged = df_merged.join(df, how="outer")
        
        df_merged.sort_index(inplace=True)
        
        # 前向填充（宏观数据更新较慢）
        df_merged.ffill(inplace=True)
        
        logger.info(f"✅ FRED数据合并完成: {len(df_merged)} 条记录, {len(df_merged.columns)} 列")
        
        return df_merged
    
    async def get_btc_related_macro(
        self,
        start_date: str = "2017-01-01",
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        获取与BTC相关的宏观指标
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            DataFrame with macro indicators
        """
        return await self.get_multiple_series(
            series_ids=BTC_RELATED_SERIES,
            start_date=start_date,
            end_date=end_date
        )
    
    def get_available_series(self) -> Dict[str, str]:
        """获取可用的指标列表"""
        return MACRO_SERIES.copy()


async def fetch_fred_macro(
    api_key: str,
    start_date: str = "2017-01-01"
) -> pd.DataFrame:
    """
    便捷函数：获取FRED宏观数据
    
    Args:
        api_key: FRED API Key
        start_date: 开始日期
        
    Returns:
        DataFrame with macro indicators
    """
    fetcher = FREDFetcher(api_key=api_key)
    
    try:
        df = await fetcher.get_btc_related_macro(start_date=start_date)
        return df
    finally:
        await fetcher.close()
