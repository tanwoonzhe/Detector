## 🚀 Linux服务器快速修复指南

### 步骤1: 检查代码是否已更新

```bash
# 运行诊断脚本
python diagnose.py
```

如果看到 ❌ 标记，说明代码未更新，请继续下一步。

### 步骤2: 更新代码

**选项A - 如果使用Git:**
```bash
git pull origin main
# 或
git fetch origin
git reset --hard origin/main
```

**选项B - 如果没有Git，手动验证关键文件:**
```bash
# 检查 hf_loader_fixed.py 第104行
grep -n "\.agg(agg_dict)" src/data_collection/hf_loader_fixed.py

# 应该看到: 104:        df_hourly = df.resample("h").agg(agg_dict)
```

如果没有看到这行，说明代码未更新。

### 步骤3: 验证修复

```bash
# 测试DataFrame重采样（不需要PyTorch）
python test_dataframe_fix.py
```

应该看到 `✅ DataFrame 重采样测试通过！`

### 步骤4: 重新训练模型

```bash
# 删除旧模型（如果存在）
rm -f models/saved/*.pth
rm -f models/saved/*.pt
rm -f data/models/*.pth

# 重新训练
python train.py --model gru
```

### 步骤5: 测试dashboard

```bash
# 启动dashboard
streamlit run app/dashboard_realtime_binance.py --server.port 8501
```

---

## 🔍 如果仍然遇到错误

### 错误1: 重采样仍然失败
```
ValueError: Data must be 1-dimensional, got ndarray of shape (66497, 3)
```

**解决方案:**
```bash
# 1. 确认文件内容
cat src/data_collection/hf_loader_fixed.py | grep -A 10 "使用agg"

# 应该看到:
# agg_dict = {
#     'open': 'first',
#     ...
# df_hourly = df.resample("h").agg(agg_dict)

# 2. 如果代码不对，重新下载或手动修改
# 找到第93-110行，替换为agg方法

# 3. 删除Python缓存
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -type f -name "*.pyc" -delete

# 4. 重启Python进程
```

### 错误2: 特征维度不匹配
```
input.size(-1) must be equal to input_size. Expected 124, got 139
```

**解决方案:**
```bash
# 1. 删除旧模型
rm -rf models/saved/*
rm -rf data/models/*

# 2. 确认main.py使用auto_build
grep "auto_build=True" main.py

# 应该看到: model.load(model_path, auto_build=True)

# 3. 重新训练模型
python train.py --model gru
```

### 错误3: 数据不足
```
⚠️ 数据不足，无法进行预测
```

**原因:** HF数据加载失败导致没有数据

**解决方案:**
```bash
# 1. 单独测试HF数据加载
python -c "
from src.data_collection.hf_loader_fixed import load_hf_btc_data
df = load_hf_btc_data()
print(f'加载成功: {len(df)} 行数据')
print(df.head())
"

# 2. 如果失败，检查错误信息并确保代码已更新
```

---

## 📝 手动修改代码（如果Git无法使用）

### 修改 src/data_collection/hf_loader_fixed.py

找到第90-110行，替换为：

```python
        # 使用agg方法进行重采样（更稳定的方法）
        print("重采样中，请稍候...")
        
        agg_dict = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last'
        }
        
        if "volume" in df.columns:
            agg_dict['volume'] = 'sum'
        
        # 使用agg一次性完成所有聚合
        df_hourly = df.resample("h").agg(agg_dict)
        
        # 如果没有volume列，添加默认值
        if "volume" not in df_hourly.columns:
            df_hourly["volume"] = 0
```

### 修改 src/models/base.py

找到 `def load(self, path: Path)` 方法（约第297行），替换为：

```python
    def load(self, path: Path, auto_build: bool = True) -> None:
        """
        加载模型
        
        Args:
            path: 模型文件路径
            auto_build: 如果模型未构建，是否自动从checkpoint中读取配置并构建
        """
        path = Path(path)
        checkpoint = torch.load(path, map_location=self.device)
        
        # 如果模型未构建，尝试自动构建
        if self.model is None:
            if auto_build and 'config' in checkpoint and 'input_shape' in checkpoint['config']:
                input_shape = checkpoint['config']['input_shape']
                n_classes = checkpoint['config'].get('n_classes', 3)
                logger.info(f"从checkpoint自动构建模型: input_shape={input_shape}, n_classes={n_classes}")
                self.build(input_shape=tuple(input_shape), n_classes=n_classes)
            else:
                raise RuntimeError("模型未构建！请先调用 build() 方法，或在checkpoint中包含input_shape信息")
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.history = checkpoint.get('history', {})
        self.config = checkpoint.get('config', {})
        self._is_trained = True
        
        logger.info(f"模型已加载: {path}")
```

---

## ✅ 验证修复成功

运行这些命令确认一切正常：

```bash
# 1. 诊断
python diagnose.py

# 2. 测试DataFrame
python test_dataframe_fix.py

# 3. 测试数据加载
python -c "from src.data_collection.hf_loader_fixed import load_hf_btc_data; df = load_hf_btc_data(); print(f'✅ 成功: {len(df)} 行')"

# 4. 训练模型
python train.py --model gru

# 5. 启动dashboard
streamlit run app/dashboard_realtime_binance.py
```

如果所有步骤都成功，问题就解决了！🎉
