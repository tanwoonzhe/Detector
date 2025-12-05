# BTC趋势预测系统

基于深度学习的加密货币趋势预测与交易信号生成系统。

## 🚀 功能特点

- **多模型集成**: GRU + Attention, BiLSTM, CNN-LSTM, LightGBM
- **多窗口预测**: 支持0.5h, 1h, 2h, 4h预测窗口
- **情感分析**: 整合Fear & Greed Index, CryptoPanic新闻, Reddit情感
- **技术分析**: 50+技术指标, 蜡烛图形态, 支撑阻力位
- **实时Dashboard**: Streamlit交互式界面
- **专业验证**: Purged K-Fold, Walk-Forward时序验证

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

2. 安装依赖:
```bash
pip install -r requirements.txt
```

3. 配置环境变量:
```bash
copy .env.example .env
# 编辑.env文件，填入API密钥(可选)
```

## 🎯 使用方法

### 训练模型

```bash
# 训练GRU模型
python main.py --train --model gru --epochs 100

# 训练所有模型
python main.py --train --model all --epochs 50
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
