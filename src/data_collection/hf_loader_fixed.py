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
        print("⚠️ 注意: 数据集较大，可能需要几分钟时间...")
        
        # 延迟导入，避免未安装时报错
        try:
            from datasets import load_dataset
        except ImportError:
            print("请先安装 datasets 库: pip install datasets")
            return pd.DataFrame()
        
        # 使用 streaming=False 避免卡在 Resolving data files
        try:
            print("加载 HuggingFace 数据集...")
            ds = load_dataset(
                "WinkingFace/CryptoLM-Bitcoin-BTC-USDT", 
                split="train",
                streaming=False
            )
            print("✅ 数据集加载成功，正在转换为 DataFrame...")
            df = ds.to_pandas()  # type: ignore
            print(f"✅ 成功加载 {len(df)} 行数据")  # type: ignore
        except Exception as e:
            print(f"❌ 加载 HuggingFace 数据集失败: {e}")
            print("建议: 使用 CoinGecko 数据训练（不加 --use-hf 参数）")
            return pd.DataFrame()
        
        # 列名映射（根据实际数据集调整）
        rename_map = {}
        for col in df.columns:  # type: ignore
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
        
        df = df.rename(columns=rename_map)  # type: ignore
        
        # 确保必要列存在
        required = {"timestamp", "open", "high", "low", "close"}
        if not required.issubset(df.columns):
            raise ValueError(f"HF数据集缺少必要列。当前列: {df.columns.tolist()}")
        
        # 时间戳处理
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.dropna(subset=["timestamp", "open", "high", "low", "close"])
        df = df.sort_values("timestamp").set_index("timestamp")
        
        # 聚合到小时级（使用 lambda 函数避免 offset 参数问题）
        print(f"正在重采样到小时级别（共 {len(df)} 行）...")
        
        # 使用更快的重采样方法
        print("重采样中，请稍候...")
        
        # 使用更简单的重采样方式
        df_resampled = df.resample("h")
        
        # 分别获取每个列的值并创建 DataFrame
        open_vals = df_resampled["open"].first()
        high_vals = df_resampled["high"].max()
        low_vals = df_resampled["low"].min()
        close_vals = df_resampled["close"].last()
        
        # 直接使用Series创建DataFrame（避免.values造成的形状问题）
        df_hourly = pd.DataFrame({
            'open': open_vals,
            'high': high_vals,
            'low': low_vals,
            'close': close_vals
        })
        
        if "volume" in df.columns:
            df_hourly["volume"] = df_resampled["volume"].sum()
        else:
            df_hourly["volume"] = 0
        
        df_hourly = df_hourly.dropna()
        
        print(f"✅ 重采样完成，得到 {len(df_hourly)} 条小时数据")
        
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
