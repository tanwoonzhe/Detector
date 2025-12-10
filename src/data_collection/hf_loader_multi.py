"""
HuggingFace 多粒度数据加载器
================================
支持将HuggingFace数据集重采样为不同时间粒度

支持的粒度:
- 1min: 原始分钟级数据
- 5min, 15min, 30min: 短期分析
- 1h, 4h: 中期分析  
- 1d: 长期分析

使用方法:
    df_15min = load_hf_btc_multi_granularity(granularity="15min")
    df_1h = load_hf_btc_multi_granularity(granularity="1h")
"""

import pandas as pd
from pathlib import Path
from typing import Optional, Literal
import logging

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

logger = logging.getLogger(__name__)

# 支持的时间粒度
VALID_GRANULARITIES = ["1min", "5min", "15min", "30min", "1h", "4h", "1d"]

# 粒度到pandas频率的映射
GRANULARITY_TO_FREQ = {
    "1min": "min",
    "5min": "5min", 
    "15min": "15min",
    "30min": "30min",
    "1h": "h",
    "4h": "4h",
    "1d": "D"
}


def load_hf_btc_multi_granularity(
    granularity: str = "1h",
    cache_dir: Optional[Path] = None,
    force_reload: bool = False,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> pd.DataFrame:
    """
    从HuggingFace数据集加载BTC历史数据，支持多种时间粒度
    
    Args:
        granularity: 时间粒度 ("1min", "5min", "15min", "30min", "1h", "4h", "1d")
        cache_dir: 缓存目录，默认为 data/hf_cache/
        force_reload: 强制重新加载（忽略缓存）
        start_date: 开始日期 (YYYY-MM-DD)
        end_date: 结束日期 (YYYY-MM-DD)
        
    Returns:
        DataFrame with columns: open, high, low, close, volume (index: timestamp)
    """
    if granularity not in VALID_GRANULARITIES:
        raise ValueError(f"无效的时间粒度: {granularity}，支持: {VALID_GRANULARITIES}")
    
    if cache_dir is None:
        cache_dir = Path(__file__).parent.parent.parent / "data" / "hf_cache"
    
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # 缓存文件路径
    cache_file = cache_dir / f"hf_btc_{granularity}.parquet"
    raw_cache = cache_dir / "hf_btc_raw.parquet"
    
    # 尝试从缓存加载
    if not force_reload and cache_file.exists():
        logger.info(f"📁 从缓存加载 {granularity} 数据: {cache_file}")
        df = pd.read_parquet(cache_file)
        
        # 日期过滤
        if start_date:
            df = df[df.index >= start_date]
        if end_date:
            df = df[df.index <= end_date]
        
        logger.info(f"   {len(df)} 条记录, {df.index.min()} ~ {df.index.max()}")
        return df
    
    # 加载原始数据
    df_raw = _load_raw_hf_data(raw_cache, force_reload)
    
    if df_raw.empty:
        return pd.DataFrame()
    
    # 重采样到目标粒度
    df_resampled = _resample_data(df_raw, granularity)
    
    # 保存缓存
    df_resampled.to_parquet(cache_file)
    logger.info(f"✅ 已缓存到: {cache_file}")
    
    # 日期过滤
    if start_date:
        df_resampled = df_resampled[df_resampled.index >= start_date]
    if end_date:
        df_resampled = df_resampled[df_resampled.index <= end_date]
    
    return df_resampled


def _load_raw_hf_data(cache_path: Path, force_reload: bool = False) -> pd.DataFrame:
    """加载原始HF数据（分钟级）"""
    
    # 检查缓存
    if not force_reload and cache_path.exists():
        logger.info(f"📁 从缓存加载原始数据: {cache_path}")
        return pd.read_parquet(cache_path)
    
    logger.info("📥 首次加载HF数据集，需要下载...")
    logger.info("⚠️ 注意: 数据集较大(约2.26M行)，可能需要几分钟...")
    
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("请先安装 datasets 库: pip install datasets")
        return pd.DataFrame()
    
    try:
        logger.info("加载 HuggingFace 数据集...")
        ds = load_dataset(
            "WinkingFace/CryptoLM-Bitcoin-BTC-USDT",
            split="train",
            streaming=False
        )
        # 处理不同的返回类型
        if hasattr(ds, 'to_pandas'):
            df: pd.DataFrame = ds.to_pandas()  # type: ignore
        else:
            # 如果是DatasetDict，取第一个split
            df = pd.DataFrame(ds)  # type: ignore
        logger.info(f"✅ 成功加载 {len(df)} 行原始数据")
        
    except Exception as e:
        logger.error(f"❌ 加载失败: {e}")
        return pd.DataFrame()
    
    # 列名映射
    rename_map = {}
    for col in df.columns:
        lc = col.lower()
        if lc in ["ts", "time", "timestamp", "date"]:
            rename_map[col] = "timestamp"
        elif lc == "open":
            rename_map[col] = "open"
        elif lc == "high":
            rename_map[col] = "high"
        elif lc == "low":
            rename_map[col] = "low"
        elif lc in ["close", "price"]:
            rename_map[col] = "close"
        elif lc in ["volume", "vol"]:
            rename_map[col] = "volume"
    
    df = df.rename(columns=rename_map)
    
    # 只保留OHLCV列
    base_cols = [c for c in ["timestamp", "open", "high", "low", "close", "volume"] if c in df.columns]
    df = df[base_cols]
    
    # 确保必要列存在
    required = {"timestamp", "open", "high", "low", "close"}
    if not required.issubset(df.columns):
        raise ValueError(f"数据集缺少必要列。当前列: {df.columns.tolist()}")
    
    # 数据清洗
    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    
    # 时间戳处理
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp", "open", "high", "low", "close"])
    df = df.sort_values("timestamp").set_index("timestamp")
    
    # 移除时区
    if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
        df.index = df.index.tz_localize(None)  # type: ignore
    
    # 保存原始缓存
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path)
    logger.info(f"✅ 原始数据已缓存: {cache_path}")
    
    return df


def _resample_data(df: pd.DataFrame, granularity: str) -> pd.DataFrame:
    """重采样数据到目标粒度"""
    
    if granularity == "1min":
        return df  # 原始数据就是分钟级
    
    freq = GRANULARITY_TO_FREQ[granularity]
    logger.info(f"📊 重采样到 {granularity} (freq={freq})...")
    
    # 计算预估组数
    if granularity == "1d":
        n_groups = int((df.index.max() - df.index.min()).days) + 1
    elif granularity == "4h":
        n_groups = int((df.index.max() - df.index.min()).total_seconds() / 14400) + 1
    elif granularity == "1h":
        n_groups = int((df.index.max() - df.index.min()).total_seconds() / 3600) + 1
    else:
        # 分钟级
        minutes = int(granularity.replace("min", ""))
        n_groups = int((df.index.max() - df.index.min()).total_seconds() / (60 * minutes)) + 1
    
    # 分组重采样
    iterator = df.groupby(pd.Grouper(freq=freq))
    if tqdm:
        iterator = tqdm(iterator, total=n_groups, desc=f"重采样到 {granularity}", unit="bar")
    
    records = []
    has_volume = "volume" in df.columns
    
    for ts, g in iterator:
        if g.empty:
            continue
        rec = {
            "timestamp": ts,
            "open": g["open"].iloc[0],
            "high": g["high"].max(),
            "low": g["low"].min(),
            "close": g["close"].iloc[-1]
        }
        if has_volume:
            rec["volume"] = g["volume"].sum()
        records.append(rec)
    
    if not records:
        logger.error("重采样结果为空")
        return pd.DataFrame()
    
    df_resampled = pd.DataFrame(records).set_index("timestamp").sort_index()
    if not has_volume:
        df_resampled["volume"] = 0
    
    df_resampled = df_resampled.dropna()
    
    logger.info(f"✅ 重采样完成: {len(df_resampled)} 条 {granularity} 数据")
    logger.info(f"   时间范围: {df_resampled.index.min()} ~ {df_resampled.index.max()}")
    
    return df_resampled


def precompute_all_granularities(cache_dir: Optional[Path] = None):
    """
    预计算所有粒度的缓存（一次性处理）
    
    这会下载原始数据并生成所有粒度的缓存文件，
    之后加载任何粒度都会非常快。
    """
    if cache_dir is None:
        cache_dir = Path(__file__).parent.parent.parent / "data" / "hf_cache"
    
    logger.info("=" * 50)
    logger.info("预计算所有时间粒度缓存")
    logger.info("=" * 50)
    
    # 先加载原始数据
    raw_cache = cache_dir / "hf_btc_raw.parquet"
    df_raw = _load_raw_hf_data(raw_cache, force_reload=False)
    
    if df_raw.empty:
        logger.error("无法加载原始数据")
        return
    
    # 生成各粒度缓存
    for granularity in VALID_GRANULARITIES:
        logger.info(f"\n处理 {granularity}...")
        cache_file = cache_dir / f"hf_btc_{granularity}.parquet"
        
        if cache_file.exists():
            logger.info(f"  已存在，跳过")
            continue
        
        df = _resample_data(df_raw, granularity)
        df.to_parquet(cache_file)
        logger.info(f"  ✅ 已保存: {cache_file}")
    
    logger.info("\n" + "=" * 50)
    logger.info("✅ 所有粒度缓存生成完成!")
    logger.info("=" * 50)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # 测试加载不同粒度
    for g in ["15min", "1h", "4h"]:
        df = load_hf_btc_multi_granularity(granularity=g)
        if not df.empty:
            print(f"\n{g}: {len(df)} 条记录")
            print(f"  范围: {df.index.min()} ~ {df.index.max()}")
