# 🔧 Dashboard 修复说明

## 问题原因

1. **Event Loop 冲突**: Streamlit 在运行时已经有一个事件循环，当使用 `asyncio.run()` 或创建新的事件循环时会产生冲突
2. **Session 绑定**: aiohttp 的 ClientSession 会绑定到特定的事件循环，当循环关闭后无法使用

## 已修复内容

### 1. Dashboard 事件循环 ✅
- 使用持久化的事件循环（`@st.cache_resource`）
- 每次请求创建新的 API 实例和 session
- 添加 `nest_asyncio` 支持嵌套事件循环
- 正确关闭 session 避免资源泄漏

### 2. HuggingFace 数据加载 ✅
- 修复 pandas `resample().agg()` 的 `first()` 和 `last()` 参数问题
- 使用 lambda 函数替代字符串参数
- 移除 `origin='start'` 参数（不需要）
- 使用小写 `'h'` 避免 FutureWarning

## 使用方法

### 1. 确保依赖已安装
```bash
pip install nest-asyncio
```

或者重新安装所有依赖：
```bash
pip install -r requirements.txt
```

### 2. 运行测试
```bash
# 测试修复是否成功
python test_dashboard_fix.py
```

### 3. 启动 Dashboard
```bash
# Binance 实时数据版本
streamlit run app/dashboard_realtime_binance.py

# 或者稳定版本（CoinGecko）
streamlit run app/dashboard_stable.py
```

### 4. 访问 Dashboard
打开浏览器访问: http://localhost:8501

## 常见问题

### Q: 仍然显示 "Event loop is closed"
**A**: 
1. 停止所有 streamlit 进程
2. 清除缓存: `streamlit cache clear`
3. 重新启动 dashboard

### Q: 数据获取失败
**A**:
1. 检查网络连接
2. Binance API 可能在某些地区受限，尝试使用 VPN
3. 或改用 `dashboard_stable.py`（使用 CoinGecko API）

### Q: HuggingFace 数据下载失败
**A**:
1. 如果在中国大陆，HuggingFace 访问困难
2. 建议使用 `--use-hf` 参数时启用代理
3. 或者不使用 `--use-hf`，只用 CoinGecko 数据训练

## 训练模型

```bash
# 只使用 CoinGecko 数据（推荐）
python train.py --model gru --epochs 100

# 使用 HuggingFace 数据（需要良好的网络）
python train.py --model gru --epochs 100 --use-hf
```

## 技术细节

### 事件循环管理
```python
@st.cache_resource
def get_event_loop():
    """获取或创建持久化的事件循环"""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop
```

### API 实例管理
```python
# 每次创建新的 API 实例（不缓存）
async def _fetch():
    api = BinancePublicAPI()
    try:
        result = await api.get_data()
        return result
    finally:
        await api.close()  # 确保关闭
```

## 更新日志

- ✅ 修复 Dashboard event loop 冲突
- ✅ 修复 HuggingFace 数据加载
- ✅ 添加 nest_asyncio 支持
- ✅ 改进错误处理和日志
- ✅ 添加测试脚本

---

**如果还有问题，请检查:**
1. Python 版本 >= 3.10
2. 所有依赖已正确安装
3. 网络连接正常
