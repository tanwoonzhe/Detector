## 🔧 重要修复说明 - 2025年12月9日

### 问题1: HF数据加载错误
**错误**: `ValueError: Data must be 1-dimensional, got ndarray of shape (66497, 3) instead`

**原因**: pandas重采样后的数据结构在不同pandas版本中可能有不同行为

**修复**: 已更新 `src/data_collection/hf_loader_fixed.py`，使用更稳定的`.agg()`方法

**重要提示**:
- ⚠️ **必须删除旧的缓存数据**: `data/raw/hf_btc_hourly.parquet`
- ⚠️ **确保代码已更新**: 检查第104行应该是 `df_hourly = df.resample("h").agg(agg_dict)`
- ⚠️ **重启Python进程**: 如果在Jupyter或IDE中运行，重启kernel

### 问题2: 模型特征维度不匹配
**错误**: `input.size(-1) must be equal to input_size. Expected 124, got 139`

**原因**: 训练时使用的特征数量与预测时不一致

**修复**: 
1. 已更新 `src/models/base.py` - 添加`auto_build`功能，自动从checkpoint读取正确的特征维度
2. 已更新 `main.py` 和 dashboard文件 - 使用`model.load(path, auto_build=True)`

**如何解决**:

#### 方式1: 使用auto_build（推荐）
```python
model = GRUPredictor()
model.load(model_path, auto_build=True)  # 自动从checkpoint读取正确配置
```

#### 方式2: 重新训练模型
如果特征工程代码已更改，建议重新训练：
```bash
python train.py --model gru
```

#### 方式3: 检查特征数量
确保训练和预测时的特征数量一致：
```python
# 检查当前特征数量
from src.features.engineer import FeatureEngineer
engineer = FeatureEngineer()
df_features = engineer.create_features(df)
feature_cols = engineer.get_feature_columns(df_features)
print(f"当前特征数: {len(feature_cols)}")
```

### 验证修复

运行测试脚本验证修复：
```bash
# 测试DataFrame重采样
python test_dataframe_fix.py

# 测试模型加载（需要先有训练好的模型）
python test_model_loading.py
```

### 文件清单
修改的文件：
- ✅ `src/data_collection/hf_loader_fixed.py` - 修复重采样方法
- ✅ `src/models/base.py` - 添加auto_build功能
- ✅ `src/models/gru.py` - 保存n_classes到config
- ✅ `main.py` - 使用auto_build加载模型
- ✅ `app/dashboard_with_prediction.py` - 使用auto_build
- ✅ `app/dashboard_realtime_binance.py` - 使用auto_build

新增的文件：
- ✅ `test_dataframe_fix.py` - 测试DataFrame重采样
- ✅ `test_model_loading.py` - 测试模型加载

### 常见问题

**Q: 我仍然看到重采样错误怎么办？**
A: 
1. 确认代码已更新（检查git status或重新拉取代码）
2. 删除缓存文件: `rm data/raw/hf_btc_hourly.parquet`
3. 重启Python进程/Jupyter kernel
4. 重新运行训练或预测

**Q: 模型加载仍然报特征维度错误？**
A: 
1. 检查是否使用了`auto_build=True`: `model.load(path, auto_build=True)`
2. 如果问题仍存在，删除旧模型并重新训练
3. 确保训练和预测时使用相同的特征工程配置

**Q: 预测失败显示"数据不足"？**
A: 
1. 确保有足够的历史数据（至少需要sequence_length条数据）
2. 检查数据加载是否成功
3. 查看日志中的详细错误信息

### 下一步

如果以上修复后问题仍然存在，请：
1. 提供完整的错误日志
2. 说明运行环境（Python版本、pandas版本、操作系统）
3. 说明使用的具体命令或脚本
