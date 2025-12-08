"""
HuggingFace数据集加载器 - 修正版
================================
修复 resample offset 参数问题
"""

import pandas as pd
from pathlib import Path
from typing import Optional


def load_hf_btc_data(cache_path: Optional[Path] = None) -> pd.DataFrame:
    """
    从HuggingFace数据集加载BTC历史数据，聚合到小时级别
    
    Args:
        cache_path: 本地缓存路径，默认为 data/hf_btc.parquet
        
    Returns:
        DataFrame with columns: open, high, low, close, volume (index: timestamp)
    """
    if cache_path is None:
        cache_path = Path(__file__).parent.parent.parent / "data" / "hf_btc.parquet"
    
    try:
        # 先尝试读缓存
        if cache_path.exists():
            print(f"从缓存加载HF数据: {cache_path}")
            return pd.read_parquet(cache_path)
        
        print("首次加载HF数据集，需要下载...")
        
        # 延迟导入，避免未安装时报错
        try:
            from datasets import load_dataset
        except ImportError:
            print("请先安装 datasets 库: pip install datasets")
            return pd.DataFrame()
        
        ds = load_dataset("WinkingFace/CryptoLM-Bitcoin-BTC-USDT", split="train")
        df = ds.to_pandas()
        
        # 列名映射（根据实际数据集调整）
        rename_map = {}
        for col in df.columns:
            lc = col.lower()
            if lc in ["ts", "time", "timestamp", "date"]:
                rename_map[col] = "timestamp"
            elif "open" in lc:
                rename_map[col] = "open"
            elif "high" in lc:
                rename_map[col] = "high"
            elif "low" in lc:
                rename_map[col] = "low"
            elif "close" in lc or lc == "price":
                rename_map[col] = "close"
            elif "volume" in lc or "vol" in lc:
                rename_map[col] = "volume"
        
        df = df.rename(columns=rename_map)
        
        # 确保必要列存在
        required = {"timestamp", "open", "high", "low", "close"}
        if not required.issubset(df.columns):
            raise ValueError(f"HF数据集缺少必要列。当前列: {df.columns.tolist()}")
        
        # 时间戳处理
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.dropna(subset=["timestamp", "open", "high", "low", "close"])
        df = df.sort_values("timestamp").set_index("timestamp")
        
        # 聚合到小时级（修复 offset 参数问题）
        agg_dict = {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
        }
        if "volume" in df.columns:
            agg_dict["volume"] = "sum"
        
        # 使用 origin='start' 替代 offset 参数
        df_hourly = df.resample("H", origin='start').agg(agg_dict).dropna()
        
        # 缓存到本地
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df_hourly.to_parquet(cache_path)
        print(f"HF数据已缓存到: {cache_path}，共 {len(df_hourly)} 条小时数据")
        
        return df_hourly
    
    except Exception as e:
        print(f"加载HF数据失败: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()


if __name__ == "__main__":
    # 测试加载
    df = load_hf_btc_data()
    if not df.empty:
        print(df.head())
        print(f"数据范围: {df.index.min()} ~ {df.index.max()}")
        print(f"总共 {len(df)} 条记录")
    else:
        print("加载失败")
