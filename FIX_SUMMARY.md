"""
修复说明文档
================================

## 问题 1: 训练时的 ValueError

### 原因
HuggingFace 数据加载成功（39492条），但错误日志显示混乱。实际问题是：
- 从截图看，数据成功加载了 39492 条记录（2017-2025年）
- 但在合并 CoinGecko 最新数据时出现时间戳比较错误
- 特征工程时初始数据显示为 2160 行（说明实际使用的是 CoinGecko 数据）

### 已修复
✅ 改进了错误日志，现在会正确显示初始/删除/剩余的数据量
✅ HuggingFace 数据加载逻辑已正确实现

### 建议
如果 HuggingFace 数据太旧（2017-2025/03），合并最新数据可能有问题。建议：
1. **只用 HuggingFace**: `python train.py --model gru --use-hf --epochs 100`
2. **只用 CoinGecko**: `python train.py --model gru --epochs 100`（默认）

不建议使用 `--merge-recent`，因为时间戳格式可能不兼容。

---

## 问题 2: Binance API 451 错误

### 原因
Binance API 返回 451 错误，可能原因：
1. **地区限制**: 某些地区无法访问 Binance API
2. **网络限制**: 防火墙或代理设置
3. **临时封锁**: 请求频率过高

### 已修复
✅ 添加了友好的错误提示
✅ 添加了超时控制（10-30秒）
✅ 提供了备用解决方案

### 解决方案

#### 方案 A: 使用 VPN
如果你在限制地区，使用 VPN 连接到允许访问的地区。

#### 方案 B: 使用 CoinGecko Dashboard（推荐）
CoinGecko 在全球都可访问：

```bash
python menu.py
# 选择 2 (启动 Dashboard)
# 选择 2 (稳定版 - CoinGecko)
```

#### 方案 C: 测试 Binance API
运行测试看是否能访问：

```bash
python test_binance_access.py
```

---

## 快速使用指南

### 训练模型

#### 使用 CoinGecko (推荐 - 稳定)
```bash
python train.py --model gru --epochs 100
```

#### 使用 HuggingFace (大量历史数据)
```bash
python train.py --model gru --epochs 100 --use-hf
```

### 启动 Dashboard

#### 方案 1: Binance 实时版（如果可访问）
```bash
streamlit run app/dashboard_realtime_binance.py
```
特点：秒级更新，1分钟/5分钟 K线

#### 方案 2: CoinGecko 稳定版（推荐）
```bash
streamlit run app/dashboard_stable.py
```
特点：小时级数据，全球可访问，稳定可靠

---

## 数据源对比

| 数据源 | 更新频率 | 历史长度 | 地区限制 | API Key | 推荐用途 |
|--------|---------|---------|---------|---------|---------|
| **CoinGecko** | 小时 | 90天 | ❌ 无 | ❌ 不需要 | ✅ 训练、稳定展示 |
| **Binance** | 秒级 | 无限 | ✅ 部分地区 | ❌ 不需要 | 实时监控 |
| **HuggingFace** | 静态 | 8年+ | ❌ 无 | ❌ 不需要 | 历史研究 |

---

## 常见问题

### Q1: 为什么 Binance Dashboard 打不开？
A: 可能是地区限制（451错误）。使用 CoinGecko Dashboard 代替：
```bash
streamlit run app/dashboard_stable.py
```

### Q2: HuggingFace 数据太旧怎么办？
A: 不要使用 `--merge-recent`，直接用 CoinGecko：
```bash
python train.py --model gru --epochs 100
```

### Q3: Dashboard 价格不更新？
A: 
- CoinGecko: 小时级数据，每小时更新一次（这是正常的）
- 如需实时数据：
  1. 检查是否能访问 Binance（运行 `python test_binance_access.py`）
  2. 如果不能，使用 VPN
  3. 或者接受小时级数据（其实对交易也够用）

### Q4: 训练时数据不足？
A: 检查 SEQUENCE_LENGTH（当前24小时）：
```python
# config/settings.py
SEQUENCE_LENGTH = 24  # 可以降到 12 或 6
```

---

## 推荐配置

### 中国大陆用户
```bash
# 训练
python train.py --model gru --epochs 100

# Dashboard
streamlit run app/dashboard_stable.py
```

### 其他地区用户
```bash
# 训练
python train.py --model gru --epochs 100 --use-hf

# Dashboard
streamlit run app/dashboard_realtime_binance.py
```

---

## 下一步

1. ✅ 测试 Binance 访问: `python test_binance_access.py`
2. ✅ 选择合适的 Dashboard 版本
3. ✅ 开始训练模型
4. ✅ 监控实时价格（如果 Binance 可用）

如有其他问题，请查看日志输出！
