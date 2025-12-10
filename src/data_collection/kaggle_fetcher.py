"""
Kaggle 数据获取器
================================
获取Kaggle上的BTC历史数据集

支持的数据集:
- mczielinski/bitcoin-historical-data (2012-2021, 分钟级)
- sudalairajkumar/cryptocurrencypricehistory (多币种日级)

使用方法:
1. 在 https://www.kaggle.com/account 创建API Token
2. 下载 kaggle.json 放到 ~/.kaggle/ 目录 (Linux/Mac) 或 C:/Users/<user>/.kaggle/ (Windows)
3. 或者设置环境变量 KAGGLE_USERNAME 和 KAGGLE_KEY
"""

import logging
from pathlib import Path
from typing import Optional, List
import pandas as pd
import os

logger = logging.getLogger(__name__)

# Kaggle数据集配置
KAGGLE_DATASETS = {
    "bitcoin_bitstamp": {
        "dataset": "mczielinski/bitcoin-historical-data",
        "file_pattern": "bitstamp*.csv",
        "description": "Bitstamp BTC/USD 分钟级数据 2012-2021"
    },
    "crypto_prices": {
        "dataset": "sudalairajkumar/cryptocurrencypricehistory",
        "file_pattern": "coin_Bitcoin.csv",
        "description": "BTC日级数据"
    }
}


class KaggleFetcher:
    """
    Kaggle 数据获取器
    
    需要先配置Kaggle API凭证
    """
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """
        初始化
        
        Args:
            cache_dir: 缓存目录，默认为 data/raw/kaggle/
        """
        if cache_dir is None:
            cache_dir = Path(__file__).parent.parent.parent / "data" / "raw" / "kaggle"
        
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._api = None
    
    def _get_api(self):
        """获取Kaggle API实例"""
        if self._api is not None:
            return self._api
        
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi  # type: ignore
            self._api = KaggleApi()
            self._api.authenticate()
            logger.info("✅ Kaggle API 认证成功")
            return self._api
        except ImportError:
            logger.error("❌ 请先安装kaggle包: pip install kaggle")
            return None
        except Exception as e:
            logger.error(f"❌ Kaggle API 认证失败: {e}")
            logger.info("请配置 ~/.kaggle/kaggle.json 或设置环境变量 KAGGLE_USERNAME/KAGGLE_KEY")
            return None
    
    def download_dataset(self, dataset_key: str = "bitcoin_bitstamp") -> Optional[Path]:
        """
        下载Kaggle数据集
        
        Args:
            dataset_key: 数据集键名
            
        Returns:
            下载的目录路径
        """
        if dataset_key not in KAGGLE_DATASETS:
            logger.error(f"未知数据集: {dataset_key}")
            logger.info(f"可用数据集: {list(KAGGLE_DATASETS.keys())}")
            return None
        
        config = KAGGLE_DATASETS[dataset_key]
        dataset_path = self.cache_dir / dataset_key
        
        # 检查是否已下载
        if dataset_path.exists() and any(dataset_path.iterdir()):
            logger.info(f"📁 使用缓存数据: {dataset_path}")
            return dataset_path
        
        api = self._get_api()
        if api is None:
            return None
        
        try:
            dataset_path.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"📥 下载Kaggle数据集: {config['dataset']}")
            logger.info(f"   {config['description']}")
            
            api.dataset_download_files(
                config["dataset"],
                path=str(dataset_path),
                unzip=True
            )
            
            logger.info(f"✅ 下载完成: {dataset_path}")
            return dataset_path
            
        except Exception as e:
            logger.error(f"❌ 下载失败: {e}")
            return None
    
    def load_bitstamp_data(
        self,
        resample_to: str = "1h",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        加载Bitstamp BTC分钟级历史数据
        
        Args:
            resample_to: 重采样频率 ("1min", "5min", "15min", "30min", "1h", "4h", "1d")
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
            
        Returns:
            DataFrame with OHLCV data
        """
        dataset_path = self.download_dataset("bitcoin_bitstamp")
        
        if dataset_path is None:
            # 尝试从本地缓存加载
            return self._try_load_local()
        
        # 查找CSV文件
        csv_files = list(dataset_path.glob("*.csv"))
        if not csv_files:
            logger.error("未找到CSV文件")
            return pd.DataFrame()
        
        # 选择最大的文件（通常是完整数据）
        csv_file = max(csv_files, key=lambda f: f.stat().st_size)
        logger.info(f"📊 加载数据文件: {csv_file.name}")
        
        try:
            # 读取CSV
            df = pd.read_csv(csv_file)
            
            # 处理时间戳
            if "Timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["Timestamp"], unit='s')
            elif "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit='s')
            else:
                logger.error("未找到时间戳列")
                return pd.DataFrame()
            
            df.set_index("timestamp", inplace=True)
            
            # 标准化列名
            column_mapping = {
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume_(BTC)": "volume",
                "Volume_(Currency)": "volume_usd",
                "Weighted_Price": "weighted_price"
            }
            df.rename(columns=column_mapping, inplace=True)
            
            # 只保留OHLCV
            ohlcv_cols = ["open", "high", "low", "close", "volume"]
            available_cols = [c for c in ohlcv_cols if c in df.columns]
            df = df[available_cols].copy()
            
            # 删除NaN和0值
            df.replace(0, pd.NA, inplace=True)
            df.dropna(inplace=True)
            
            # 日期过滤
            if start_date:
                df = df[df.index >= start_date]
            if end_date:
                df = df[df.index <= end_date]
            
            # 重采样
            if resample_to != "1min":
                logger.info(f"📊 重采样到 {resample_to}...")
                df = df.resample(resample_to).agg({
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                    "volume": "sum"
                }).dropna()
            
            logger.info(f"✅ Kaggle数据加载完成: {len(df)} 条记录")
            logger.info(f"   时间范围: {df.index.min()} ~ {df.index.max()}")
            
            return df
            
        except Exception as e:
            logger.error(f"❌ 加载数据失败: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()
    
    def _try_load_local(self) -> pd.DataFrame:
        """尝试从本地缓存加载"""
        for key in KAGGLE_DATASETS:
            path = self.cache_dir / key
            if path.exists():
                csv_files = list(path.glob("*.csv"))
                if csv_files:
                    logger.info(f"📁 从本地缓存加载: {csv_files[0]}")
                    try:
                        return pd.read_csv(csv_files[0])
                    except Exception:
                        pass
        return pd.DataFrame()
    
    def get_available_datasets(self) -> List[str]:
        """获取可用数据集列表"""
        return list(KAGGLE_DATASETS.keys())


def load_kaggle_btc(
    resample_to: str = "1h",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> pd.DataFrame:
    """
    便捷函数：加载Kaggle BTC数据
    
    Args:
        resample_to: 重采样频率
        start_date: 开始日期
        end_date: 结束日期
        
    Returns:
        DataFrame with OHLCV data
    """
    fetcher = KaggleFetcher()
    return fetcher.load_bitstamp_data(
        resample_to=resample_to,
        start_date=start_date,
        end_date=end_date
    )
