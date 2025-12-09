"""
HuggingFace数据集加载器 - 修正版
================================
修复 resample offset 参数问题
"""

import pandas as pd
from pathlib import Path
from typing import Optional

try:
    from tqdm.auto import tqdm
except ImportError:  # tqdm 非必须，没有则降级为无进度条
    tqdm = None


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
        # 精确列名映射，避免将 BL_Lower / MN_Lower 误映射到 low
        rename_map = {}
        for col in df.columns:  # type: ignore
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
        
        df = df.rename(columns=rename_map)  # type: ignore

        # 只保留基础OHLCV列，防止重复/多维列干扰
        base_cols = [c for c in ["timestamp", "open", "high", "low", "close", "volume"] if c in df.columns]
        df = df[base_cols]

        # 确保必要列存在
        required = {"timestamp", "open", "high", "low", "close"}
        if not required.issubset(df.columns):
            raise ValueError(f"HF数据集缺少必要列。当前列: {df.columns.tolist()}")

        # 转换数值列为浮点，过滤异常字符串/对象，防止聚合后写 parquet 失败
        def _flatten_column(col_data):
            """将可能的二维/列表/对象列展开为一维Series"""
            import numpy as np
            # 优先直接拿底层 ndarray
            if hasattr(col_data, "to_numpy"):
                arr = col_data.to_numpy()
            else:
                arr = np.array(col_data)
            # 如果是二维 (N, k)，取第0列
            if getattr(arr, "ndim", 1) > 1:
                arr = arr[:, 0]
            return pd.Series(arr)

        def _unwrap_scalar(x):
            if isinstance(x, (list, tuple)):
                return x[0] if len(x) else None
            if hasattr(x, "shape") and getattr(x, "ndim", 1) > 0:
                try:
                    return x.item()
                except Exception:
                    try:
                        return x[0] if len(x) else None
                    except Exception:
                        return None
            return x

        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns:
                col_series = _flatten_column(df[col]).apply(_unwrap_scalar)
                df[col] = pd.to_numeric(col_series.astype(str), errors="coerce")
        
        # 时间戳处理
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.dropna(subset=["timestamp", "open", "high", "low", "close"])
        df = df.sort_values("timestamp").set_index("timestamp")
        
        # 聚合到小时级（使用 lambda 函数避免 offset 参数问题）
        print(f"正在重采样到小时级别（共 {len(df)} 行）...")
        
        # 使用更快的重采样方法
        print("重采样中，请稍候...")
        
        # 分组重采样并显示进度条（兼容 pandas 2.3.x）
        # 计算预估小时数用于进度条
        n_hours = int((df.index.max() - df.index.min()).total_seconds() // 3600) + 1
        iterator = df.groupby(pd.Grouper(freq="h"))
        if tqdm:
            iterator = tqdm(iterator, total=n_hours, desc="重采样中", unit="hour")
        
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
            raise ValueError("重采样结果为空，请检查源数据")
        
        df_hourly = pd.DataFrame(records).set_index("timestamp").sort_index()
        if not has_volume:
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
