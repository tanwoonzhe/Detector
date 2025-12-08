# 🔧 问题修复总结

## ✅ 已修复的问题

### 问题1: Dashboard 显示后变空白
**原因**: `time.sleep(1) + st.rerun()` 造成无限重载循环

**解决方案**:
1. 创建 **`dashboard_stable.py`** (推荐使用)
2. 使用 `@st.cache_data(ttl=15)` 缓存数据
3. 改进刷新逻辑，避免无限循环
4. 添加手动刷新按钮

**使用方法**:
```powershell
streamlit run app/dashboard_stable.py
```

---

### 问题2: 训练时 "特征工程后数据为空"
**原因**: 
- CoinGecko 返回数据不足（<100行）
- 特征工程需要大量滚动窗口计算（最大72小时）
- 计算后产生大量 NaN，被 dropna() 删除

**解决方案**:
1. 修改 `train.py` 支持 `--use-hf` 参数（HuggingFace 数据集）
2. 增强 `engineer.py` 错误日志，显示 NaN 统计
3. 提供数据诊断脚本 `test_data_fetch.py`

**使用方法**:
```powershell
# 诊断数据量
python test_data_fetch.py

# 方案1: 使用 HuggingFace 数据（推荐）
python train.py --model gru --epochs 100 --use-hf

# 方案2: 混合数据源
python train.py --model all --epochs 200 --use-hf --merge-recent
```

---

## 📋 文件清单

### 新创建的文件
1. **`app/dashboard_stable.py`** ⭐ - 稳定版 Dashboard（解决空白问题）
2. **`test_data_fetch.py`** - 数据量诊断工具
3. **`TROUBLESHOOTING.md`** - 本文件

### 修改的文件
1. **`train.py`** - 添加 HF 数据支持，修复 fetch_data()
2. **`src/features/engineer.py`** - 增强错误日志
3. **`menu.py`** - 更新 Dashboard 选项
4. **`USAGE_GUIDE.md`** - 更新使用指南

---

## 🚀 快速启动

### 启动 Dashboard（推荐）
```powershell
python menu.py
# 选择: 2 → 1 (稳定版)
```

或直接：
```powershell
streamlit run app/dashboard_stable.py
```

### 训练模型

#### 步骤1: 诊断数据
```powershell
python test_data_fetch.py
```

#### 步骤2a: 如果 CoinGecko 数据充足（>100行）
```powershell
python train.py --model gru --epochs 100
```

#### 步骤2b: 如果数据不足，使用 HF 数据集
```powershell
# 首次需要下载数据集（可能需要几分钟）
python menu.py
# 选择: 4 (加载 HuggingFace 数据集)

# 然后训练
python train.py --model gru --epochs 100 --use-hf
```

---

## 🔍 诊断工具

### 1. 测试 CoinGecko API
```powershell
python test_coingecko.py
```

### 2. 测试数据量
```powershell
python test_data_fetch.py
```

### 3. 测试训练参数
```powershell
python test_train_args.py --model all --epochs 200 --use-hf
```

---

## ⚠️ 常见错误及解决

### 错误1: `ValueError: 特征工程后数据为空`
```
解决: 使用 HF 数据集
python train.py --model gru --epochs 100 --use-hf
```

### 错误2: Dashboard 空白
```
解决: 使用稳定版
streamlit run app/dashboard_stable.py
```

### 错误3: `OSError: [WinError 1114] DLL initialization failed`
```
原因: 本地 PyTorch 环境问题
解决: 在 Vast.ai 上运行（已配置好环境）
```

### 错误4: `401 Unauthorized` (CoinGecko)
```
原因: API 参数错误或限流
状态: 已修复（移除 interval 参数）
```

---

## 📊 Dashboard 功能对比

| 功能 | 稳定版 | 完整版 | 简化版 |
|------|--------|--------|--------|
| 真实价格 | ✅ | ✅ | ✅ |
| 自动刷新 | ✅ 可配置 | ❌ | ❌ |
| 手动刷新 | ✅ | ❌ | ❌ |
| 防空白 | ✅ | ⚠️ | ⚠️ |
| 技术指标 | ✅ | ✅ | ✅ |
| 智能信号 | ✅ | ❌ 随机 | ❌ 随机 |
| 侧边栏选项 | ✅ 完整 | ✅ 完整 | ⚠️ 简化 |

**推荐**: 稳定版 (`dashboard_stable.py`)

---

## 🌐 Vast.ai 部署步骤

### 1. SSH 连接
```powershell
ssh -p 22524 root@58.242.92.47
```

### 2. 进入项目目录
```bash
cd /workspace/Detector
```

### 3. 测试数据
```bash
python test_data_fetch.py
```

### 4. 加载 HF 数据（首次）
```bash
python menu.py
# 选择 4
```

### 5. 训练模型
```bash
python train.py --model all --epochs 200 --use-hf --merge-recent
```

### 6. 启动 Dashboard（带端口转发）
本地执行：
```powershell
ssh -p 22524 -N -T -L 8501:localhost:8501 root@58.242.92.47
```

服务器执行：
```bash
streamlit run app/dashboard_stable.py
```

浏览器打开: `http://localhost:8501`

---

## 📈 预期结果

### Dashboard 正常运行
- ✅ 显示真实 BTC 价格（~$91,000）
- ✅ 每 15 秒自动刷新（可调整）
- ✅ 交易信号基于技术指标
- ✅ 置信度动态变化
- ✅ 不会变空白

### 训练正常完成
- ✅ 数据获取: 2160+ 条记录（CoinGecko）或更多（HF）
- ✅ 特征工程: 保留 >100 行数据
- ✅ 训练完成: 保存模型到 `models/saved/`
- ✅ 验证准确率: >90%

---

## 🆘 仍然有问题？

### 检查清单
- [ ] 运行 `python test_data_fetch.py` 确认数据量
- [ ] 运行 `python test_coingecko.py` 确认 API 连接
- [ ] 使用 `dashboard_stable.py` 而非其他版本
- [ ] 训练时加上 `--use-hf` 参数
- [ ] 检查网络连接和防火墙设置

### 日志输出
查看详细日志以诊断问题：
```bash
# 训练日志会显示每步的数据量
python train.py --model gru --epochs 100 --use-hf 2>&1 | tee train.log
```

---

**最后更新**: 2025-12-08  
**版本**: v2.1 (稳定版修复)
