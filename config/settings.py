"""
BTCUSDT 预测系统配置文件
包含所有可配置参数：API密钥、模型参数、交易阈值等
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# ===== 路径配置 =====
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = DATA_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"

# 创建目录
for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, LOGS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# ===== 数据库配置 =====
DATABASE_URL = f"sqlite:///{DATA_DIR}/btcusdt_cache.db"

# ===== API配置 =====
class APIConfig:
    # CoinGecko (免费版)
    COINGECKO_BASE_URL = "https://api.coingecko.com/api/v3"
    COINGECKO_RATE_LIMIT = 25  # 每分钟请求数 (保守值)
    
    # Binance (预留)
    BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "")
    BINANCE_SECRET_KEY = os.getenv("BINANCE_SECRET_KEY", "")
    BINANCE_BASE_URL = "https://api.binance.com/api/v3"
    
    # Fear & Greed Index
    FEAR_GREED_URL = "https://api.alternative.me/fng/"
    
    # CryptoPanic
    CRYPTOPANIC_API_KEY = os.getenv("CRYPTOPANIC_API_KEY", "")
    CRYPTOPANIC_BASE_URL = "https://cryptopanic.com/api/v1"
    
    # Reddit
    REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID", "")
    REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET", "")
    REDDIT_USER_AGENT = "BTCUSDT_Predictor/1.0"

# ===== 交易对配置 =====
class TradingConfig:
    SYMBOL = "bitcoin"  # CoinGecko ID
    SYMBOL_BINANCE = "BTCUSDT"  # Binance符号
    VS_CURRENCY = "usd"
    
    # 预测时间窗口（小时）
    PREDICTION_WINDOWS = [0.5, 1, 2, 4]
    
    # 横盘阈值
    SIDEWAYS_THRESHOLD = 0.005  # ±0.5%
    
    # 数据获取天数（CoinGecko免费版限制90天）
    HISTORY_DAYS = 90

# ===== 模型配置 =====
class ModelConfig:
    # 序列长度（使用多少小时历史数据）
    SEQUENCE_LENGTH = 24  # 1天 = 24小时 (针对90天数据优化)
    
    # 通用 Dropout
    DROPOUT = 0.2
    
    # GRU模型参数
    GRU_HIDDEN_SIZE = 128
    GRU_NUM_LAYERS = 2
    GRU_DROPOUT = 0.2
    
    # BiLSTM模型参数
    BILSTM_HIDDEN_SIZE = 64
    BILSTM_NUM_LAYERS = 2
    BILSTM_DROPOUT = 0.2
    
    # LSTM模型参数
    LSTM_HIDDEN_SIZE = 64
    LSTM_NUM_LAYERS = 2
    
    # CNN-LSTM模型参数
    CNN_FILTERS = 64
    CNN_KERNEL_SIZE = 3
    
    # 训练参数
    BATCH_SIZE = 32  # GTX 1650优化
    LEARNING_RATE = 0.001
    EPOCHS = 100
    EARLY_STOPPING_PATIENCE = 15
    
    # 集成权重
    ENSEMBLE_WEIGHTS = {
        "gru": 0.4,
        "bilstm": 0.3,
        "cnn_lstm": 0.2,
        "lightgbm": 0.1
    }
    
    # 设备配置
    DEVICE = "cuda"  # GTX 1650

# ===== 特征配置 =====
class FeatureConfig:
    # 技术指标周期
    RSI_PERIOD = 14
    MACD_FAST = 12
    MACD_SLOW = 26
    MACD_SIGNAL = 9
    BB_PERIOD = 20
    BB_STD = 2
    
    # 移动平均周期（减小最大窗口从 50 到 30 以保留更多数据）
    SMA_PERIODS = [7, 14, 30]  # 原来 [7, 21, 50]
    EMA_PERIODS = [12, 26]
    
    # 收益率回溯周期（小时）- 减小最大周期
    RETURN_PERIODS = [1, 2, 4, 6, 12]  # 原来 [1, 2, 4, 6, 12, 24]
    
    # 技术指标滚动窗口（用于多时间框架分析）
    TECHNICAL_WINDOWS = [5, 10, 20, 30, 50]  # 最大50小时，避免数据损失

# ===== 情感分析配置 =====
class SentimentConfig:
    # CryptoBERT模型
    CRYPTOBERT_MODEL = "ElKulako/cryptobert"
    
    # VADER用于社交媒体
    USE_VADER = True
    
    # 情感聚合权重
    SENTIMENT_WEIGHTS = {
        "fear_greed": 0.3,
        "news": 0.4,
        "reddit": 0.3
    }
    
    # 新闻获取数量
    NEWS_LIMIT = 50
    
    # Reddit帖子数量
    REDDIT_LIMIT = 100

# ===== 信号配置 =====
class SignalConfig:
    # 最小置信度
    MIN_CONFIDENCE = 0.5
    
    # 信号级别阈值
    STRONG_BUY_THRESHOLD = 0.8
    BUY_THRESHOLD = 0.6
    SELL_THRESHOLD = 0.4
    STRONG_SELL_THRESHOLD = 0.2
    
    # 信号名称
    SIGNAL_NAMES = {
        5: "强烈买入",
        4: "买入",
        3: "持有",
        2: "卖出",
        1: "强烈卖出"
    }

# ===== 定时任务配置 =====
class SchedulerConfig:
    # 数据刷新间隔（分钟）
    DATA_REFRESH_INTERVAL = 30
    
    # 模型重训练间隔（小时）
    MODEL_RETRAIN_INTERVAL = 24

# ===== 日志配置 =====
class LogConfig:
    LOG_LEVEL = "INFO"
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_FILE = LOGS_DIR / "btcusdt_predictor.log"
