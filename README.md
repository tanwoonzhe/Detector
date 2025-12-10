# BTC趋势预测系统

基于深度学习的加密货币趋势预测与交易信号生成系统。

## 🚀 功能特点

- **多模型集成**: GRU + Attention, BiLSTM, CNN-LSTM, LightGBM
- **多窗口预测**: 支持0.5h, 1h, 2h, 4h预测窗口
- **多数据源支持**: CoinGecko, FMP, CoinMetrics, HuggingFace
- **情感分析**: 整合Fear & Greed Index, CryptoPanic新闻, Reddit情感
- **技术分析**: 50+技术指标, 蜡烛图形态, 支撑阻力位
- **宏观数据**: 国债收益率, VIX, S&P500, 黄金, 美元指数
- **链上数据**: 活跃地址, 哈希率, NVT, 交易数等
- **实时Dashboard**: Streamlit交互式界面
- **专业验证**: Purged K-Fold, Walk-Forward时序验证

## 📊 数据源

### 支持的数据源一览

| 数据类别 | 数据源 | 时间范围 | 粒度 | 说明 |
|---------|--------|---------|------|------|
| BTC价格 | HuggingFace | 2017-2025 | 1min~1d | 完整历史数据(推荐) |
| BTC价格 | Binance历史归档 | 2017-今 | 1m~1d | 官方数据源 |
| BTC价格 | Kaggle | 2012-2024 | 1min~1d | Bitstamp数据 |
| BTC价格 | CoinGecko | 90天 | 小时级 | 实时数据 |
| BTC价格 | FMP | 多年 | 小时级 | 付费API |
| 宏观经济 | FRED | 50+年 | 日级 | 利率/通胀/M2 |
| 宏观经济 | FMP | 多年 | 日级 | VIX/股指/商品 |
| 链上数据 | CoinMetrics | 2011-今 | 日级 | 活跃地址/哈希率/NVT |
| 新闻情绪 | FMP/CryptoPanic | - | - | 加密货币新闻

## 📁 项目结构

```
Detect/
├── config/                 # 配置文件
│   ├── __init__.py
│   └── settings.py         # 所有配置参数
├── src/
│   ├── data_collection/    # 数据采集
│   │   ├── base.py         # 抽象基类
│   │   ├── coingecko_fetcher.py  # CoinGecko数据源
│   │   ├── fmp_fetcher.py        # FMP数据源（宏观+加密）
│   │   ├── coinmetrics_fetcher.py # CoinMetrics链上数据
│   │   ├── data_pipeline.py      # 多源数据合并管道
│   │   ├── binance_fetcher.py    # Binance数据源(备用)
│   │   └── cache.py        # SQLite缓存
│   ├── sentiment/          # 情感分析
│   │   ├── sources/        # 情感数据源
│   │   ├── analyzer.py     # CryptoBERT + VADER分析器
│   │   └── aggregator.py   # 多源聚合
│   ├── features/           # 特征工程
│   │   ├── technical.py    # 技术指标
│   │   ├── patterns.py     # 蜡烛图形态
│   │   ├── support_resistance.py  # 支撑阻力
│   │   └── engineer.py     # 特征工程主模块
│   ├── validation/         # 验证框架
│   │   └── time_series.py  # Purged K-Fold, Walk-Forward
│   ├── models/             # 预测模型
│   │   ├── base.py         # PyTorch基类
│   │   ├── gru.py          # GRU + Attention
│   │   ├── bilstm.py       # BiLSTM
│   │   ├── cnn_lstm.py     # CNN-LSTM混合
│   │   ├── lightgbm_model.py  # LightGBM基准
│   │   └── ensemble.py     # 模型集成
│   └── signals/            # 信号生成
│       └── generator.py    # 交易信号生成器
├── app/
│   └── dashboard.py        # Streamlit界面
├── main.py                 # 主入口
├── train.py                # 训练脚本
├── requirements.txt        # 依赖
└── .env.example            # 环境变量模板
```

## 🔧 安装

1. 创建虚拟环境:
```bash
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac
```

ssh -p 22524 -L 8501:localhost:8501 root@58.242.92.4

2. 安装依赖:
```bash
pip install -r requirements.txt
```

3. 配置环境变量:
```bash
copy .env.example .env
# 编辑.env文件，填入API密钥
```

### 环境变量说明

```env
# Financial Modeling Prep (推荐，支持宏观+加密数据)
FMP_API_KEY=your_fmp_api_key

# CoinMetrics (链上数据，社区版免费)
COINMETRICS_API_KEY=  # 可为空

# 其他可选
BINANCE_API_KEY=your_binance_key
BINANCE_SECRET_KEY=your_binance_secret
CRYPTOPANIC_API_KEY=your_cryptopanic_key
```

## 🎯 使用方法

### 快速开始（推荐）

```bash
# 使用交互式菜单
python menu.py
```

### 训练模型

```bash
# 基础训练（使用CoinGecko）
python train.py --model gru --epochs 100

# 使用FMP数据
python train.py --model gru --use-fmp --fmp-days 90

# 🌟 使用多数据源管道（推荐）
python train.py --model all --use-pipeline --fmp-days 90

# 多源管道 + 自定义选项
python train.py --model gru --use-pipeline --no-macro  # 不含宏观数据
python train.py --model gru --use-pipeline --no-onchain  # 不含链上数据

# 使用HuggingFace历史数据
python train.py --model all --use-hf --merge-recent
```

### 数据源选项

| 参数 | 说明 |
|------|------|
| **长历史数据源** | |
| `--use-hf-multi` | 使用多粒度HuggingFace数据(2017-2025) |
| `--use-binance-hist` | 使用Binance历史归档数据(2017-今) |
| `--use-kaggle` | 使用Kaggle历史数据(2012-2024) |
| `--interval` | 数据粒度: 1min/5min/15min/30min/1h/4h/1d |
| `--days N` | 获取N天历史数据 |
| **传统数据源** | |
| `--use-pipeline` | 使用多数据源管道（合并宏观+链上+跨市场） |
| `--use-fmp` | 使用FMP获取BTC数据 |
| `--use-hf` | 使用HuggingFace历史数据集（小时级） |
| `--fmp-days N` | FMP数据天数 |
| `--no-macro` | 不包含宏观经济数据 |
| `--no-onchain` | 不包含链上数据 |
| `--merge-recent` | 合并最新CoinGecko数据 |

### 训练示例

```bash
# 基础训练（使用CoinGecko 90天数据）
python train.py --model gru --epochs 100

# 🌟 使用HuggingFace多粒度数据（推荐）
python train.py --model cnn_lstm --use-hf-multi --interval 15min --epochs 100

# 🌟 使用Binance历史归档（官方数据，最准确）
python train.py --model all --use-binance-hist --interval 1h --days 365

# 使用Kaggle历史数据
python train.py --model gru --use-kaggle --epochs 100

# 使用FMP数据
python train.py --model gru --use-fmp --fmp-days 90

# 使用多数据源管道（合并宏观+链上）
python train.py --model all --use-pipeline --fmp-days 90

# 多源管道 + 自定义选项
python train.py --model gru --use-pipeline --no-macro  # 不含宏观数据
python train.py --model gru --use-pipeline --no-onchain  # 不含链上数据

# 使用HuggingFace历史数据 + 最新数据
python train.py --model all --use-hf --merge-recent
```

### 测试训练流程

```bash
# 测试数据和特征工程是否正常
python test_training_pipeline.py
```

### 启动Dashboard

```bash
python main.py --dashboard
```

然后在浏览器访问: http://localhost:8501

### 单次预测

```bash
python main.py --predict
```

## 📊 模型说明

### GRU + Attention (主模型)
- 针对GTX 1650优化 (4GB VRAM)
- 2层GRU, 128维隐藏层
- 自注意力机制捕获关键时间点
- 参数量: ~300K

### BiLSTM
- 双向LSTM捕获前后文信息
- 适合中长期趋势预测

### CNN-LSTM
- 1D CNN提取局部模式
- LSTM捕获长期依赖
- 多尺度特征提取

### LightGBM
- 快速基准模型
- 特征重要性分析
- 不需要GPU

## 📈 技术指标

- **趋势**: SMA, EMA, MACD, ADX
- **动量**: RSI, Stochastic, ROC
- **波动率**: Bollinger Bands, ATR
- **成交量**: OBV, VWAP
- **形态**: 十字星, 锤子线, 吞没形态等

## 🎲 预测标签

- **0 (下跌)**: 预测收益 < -0.5%
- **1 (横盘)**: 预测收益在 ±0.5% 之间
- **2 (上涨)**: 预测收益 > +0.5%

## ⚠️ 免责声明

本系统仅供学习和研究使用，不构成任何投资建议。加密货币市场风险极高，请谨慎投资。

## 📚 参考资料

- López de Prado《Advances in Financial Machine Learning》
- Cho et al., "Learning Phrase Representations using RNN Encoder-Decoder"
- CSDN/Kaggle金融时序预测最佳实践

## 📝 License

MIT License
