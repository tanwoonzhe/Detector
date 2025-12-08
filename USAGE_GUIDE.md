# 🚀 BTC趋势预测系统 - 使用指南

## 📋 问题修复说明

### ✅ 问题1: 训练命令参数错误
**原因**: `train.py` 缺少 `--use-hf` 和 `--merge-recent` 参数定义

**修复**: 
- 添加了 `--use-hf` 参数（使用 HuggingFace 历史数据）
- 添加了 `--merge-recent` 参数（合并最新 CoinGecko 数据）

**测试**:
```powershell
python test_train_args.py --model all --epochs 200 --use-hf --merge-recent
```

### ✅ 问题2: Dashboard 价格不自动刷新
**原因**: 
- 旧版 dashboard 是静态页面，需要手动刷新
- 预测信号是随机生成的演示数据

**修复**:
创建了 **实时自动刷新版 Dashboard** (`dashboard_realtime.py`)

**新功能**:
- ⏱️ 每 15 秒自动刷新价格和信号（可调整 5-60 秒）
- 📊 基于真实技术指标计算交易信号（RSI、SMA、布林带、价格动量）
- 🔄 智能信号算法，不再是随机数
- 📈 实时置信度计算
- 🕐 显示最后更新时间和倒计时

---

## 🎯 快速开始

### 方式1: 使用交互式菜单 (推荐)
```powershell
python menu.py
```

### 方式2: 直接启动实时 Dashboard
```powershell
streamlit run app/dashboard_realtime.py
```

### 方式3: 命令行训练
```powershell
# 使用 CoinGecko 数据训练
python train.py --model gru --epochs 100

# 使用 HuggingFace 历史数据
python train.py --model all --epochs 200 --use-hf

# 混合数据源
python train.py --model all --epochs 200 --use-hf --merge-recent
```

---

## 📊 Dashboard 版本对比

| 版本 | 文件名 | 特点 | 推荐场景 |
|------|--------|------|---------|
| **稳定版** ⭐ | `dashboard_stable.py` | 缓存机制，无空白问题，手动+自动刷新 | **首选** |
| 完整版 | `dashboard_complete.py` | 所有侧边栏选项，手动刷新 | 详细分析 |
| 简化版 | `dashboard_fixed.py` | 轻量快速，真实价格 | 快速测试 |
| 实时版 | `dashboard_realtime.py` | 旧实时版（有空白bug） | 不推荐 |

---

## ⚙️ 实时 Dashboard 配置

启动后在侧边栏可以调整：

1. **自动刷新间隔**: 5-60 秒（默认 15 秒）
2. **显示选项**:
   - 技术指标详情
   - 市场情感仪表盘

---

## 🔍 信号计算逻辑

实时 Dashboard 的交易信号基于多个技术指标综合计算：

### 买入信号 (BUY)
- RSI < 30 (超卖)
- SMA 24 > SMA 72 (金叉)
- 价格接近布林带下轨
- 价格上涨动量 > 1%

### 卖出信号 (SELL)
- RSI > 70 (超买)
- SMA 24 < SMA 72 (死叉)
- 价格接近布林带上轨
- 价格下跌动量 > 1%

### 观望信号 (HOLD)
- 指标信号混合或中性

**置信度计算**: 基于所有指标的一致性，范围 50%-95%

---

## 🌐 Vast.ai 部署

### SSH 连接
```powershell
ssh -p 22524 root@58.242.92.47
```

### 启动实时 Dashboard（带端口转发）
```powershell
# 本地执行
ssh -p 22524 -N -T -L 8501:localhost:8501 root@58.242.92.47
```

然后在浏览器打开: `http://localhost:8501`

或者使用 Vast.ai 的 Tunnels 功能直接暴露公网地址。

---

## 📦 HuggingFace 数据集

### 首次加载
```powershell
python menu.py
# 选择: 4. 加载 HuggingFace 数据集
```

### 数据缓存位置
`data/hf_btc.parquet` (自动创建)

### 数据集信息
- **来源**: WinkingFace/CryptoLM-Bitcoin-BTC-USDT
- **粒度**: 小时级 OHLCV
- **范围**: 历史全量数据（具体看数据集）

---

## 🐛 常见问题

### Q1: 本地 PyTorch DLL 错误
**A**: 这是本地环境问题，在 Vast.ai 上不会出现。如需本地运行，尝试：
```powershell
pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### Q2: Dashboard 显示 "无法获取数据"
**A**: 检查网络连接，CoinGecko API 可能限流（免费版有限制）

### Q3: 自动刷新太快导致卡顿
**A**: 在侧边栏调整刷新间隔到 30-60 秒

### Q4: 信号一直是 HOLD
**A**: 市场可能处于横盘期，或数据不足计算技术指标（需要至少 72 小时数据）

### Q5: 训练时出现 "特征工程后数据为空"
**A**: CoinGecko 数据不足（<100行），解决方法：
```powershell
# 方法1: 测试数据量
python test_data_fetch.py

# 方法2: 使用 HuggingFace 数据集
python train.py --model gru --epochs 100 --use-hf

# 方法3: 增加天数（但 CoinGecko 免费版限制90天）
```

### Q6: Dashboard 显示后变空白
**A**: 使用稳定版 dashboard，已修复无限重载问题：
```powershell
streamlit run app/dashboard_stable.py
```

---

## 📊 性能优化建议

### Vast.ai 实例配置
- GPU: RTX 4070 或以上
- 内存: 16GB+
- 存储: 20GB+（用于缓存数据集）

### Dashboard 优化
- 如果数据量大，调整 `days=7` 为 `days=3` 减少加载时间
- 关闭不需要的侧边栏选项
- 增加刷新间隔到 30-60 秒

---

## 🎓 下一步

1. ✅ 训练模型（使用混合数据源获得最佳效果）
2. ✅ 启动实时 Dashboard 观察信号
3. 🔄 根据历史表现调整信号阈值
4. 📈 部署到生产环境（Vast.ai + 公网访问）

---

## 📞 支持

如有问题，检查：
1. `test_coingecko.py` - 测试 API 连接
2. `test_train_args.py` - 测试训练参数
3. 终端日志输出

---

**最后更新**: 2025-12-08
**版本**: v2.0 (实时刷新版)
