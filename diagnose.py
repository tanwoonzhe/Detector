#!/usr/bin/env python3
"""
快速诊断脚本
============
检查修复是否正确应用
"""
import sys
from pathlib import Path

print("=" * 70)
print("🔍 诊断检查 - 修复状态")
print("=" * 70)

# 1. 检查文件是否存在
print("\n1️⃣  检查关键文件...")
files_to_check = [
    "src/data_collection/hf_loader_fixed.py",
    "src/data_collection/hf_loader.py",
    "src/models/base.py",
    "src/models/gru.py",
    "main.py",
]

for file in files_to_check:
    filepath = Path(file)
    if filepath.exists():
        print(f"  ✅ {file}")
    else:
        print(f"  ❌ {file} 不存在！")

# 2. 检查关键代码是否已更新
print("\n2️⃣  检查代码是否已更新...")

checks = [
    {
        "file": "src/data_collection/hf_loader_fixed.py",
        "search": ".agg(agg_dict)",
        "desc": "HF loader使用agg方法"
    },
    {
        "file": "src/data_collection/hf_loader.py",
        "search": ".agg(agg_dict)",
        "desc": "HF loader(旧)使用agg方法"
    },
    {
        "file": "src/models/base.py",
        "search": "auto_build",
        "desc": "模型加载支持auto_build"
    },
    {
        "file": "main.py",
        "search": "auto_build=True",
        "desc": "main.py使用auto_build"
    }
]

for check in checks:
    filepath = Path(check["file"])
    if not filepath.exists():
        print(f"  ⚠️  {check['desc']}: 文件不存在")
        continue
    
    content = filepath.read_text(encoding='utf-8')
    if check["search"] in content:
        print(f"  ✅ {check['desc']}")
    else:
        print(f"  ❌ {check['desc']} - 代码未更新！")

# 3. 检查缓存文件
print("\n3️⃣  检查缓存文件...")
cache_files = [
    "data/raw/hf_btc_hourly.parquet",
    "data/raw/hf_btc_hourly.csv",
]

any_cache = False
for cache_file in cache_files:
    filepath = Path(cache_file)
    if filepath.exists():
        print(f"  ⚠️  发现缓存: {cache_file}")
        print(f"      大小: {filepath.stat().st_size / 1024 / 1024:.2f} MB")
        print(f"      建议: 删除此文件以强制重新加载数据")
        any_cache = True

if not any_cache:
    print("  ✅ 没有发现旧缓存文件")

# 4. 检查模型文件
print("\n4️⃣  检查模型文件...")
model_dirs = [
    "models/saved",
    "data/models",
]

any_model = False
for model_dir in model_dirs:
    dirpath = Path(model_dir)
    if dirpath.exists():
        models = list(dirpath.glob("*.pt")) + list(dirpath.glob("*.pth"))
        if models:
            print(f"  📦 {model_dir}:")
            for model in models:
                size_mb = model.stat().st_size / 1024 / 1024
                print(f"      - {model.name} ({size_mb:.2f} MB)")
                any_model = True

if not any_model:
    print("  ℹ️  没有发现训练好的模型（需要先训练）")

# 5. 环境检查
print("\n5️⃣  环境信息...")
print(f"  Python: {sys.version.split()[0]}")
print(f"  平台: {sys.platform}")

try:
    import pandas as pd
    print(f"  Pandas: {pd.__version__}")
except:
    print("  Pandas: 未安装")

try:
    import torch
    print(f"  PyTorch: {torch.__version__}")
except:
    print("  PyTorch: 未安装或无法加载")

# 总结
print("\n" + "=" * 70)
print("📋 诊断总结")
print("=" * 70)

print("""
如果看到 ❌ 标记：
  → 代码可能未正确更新，请重新拉取代码

如果看到 ⚠️  缓存警告：
  → 删除缓存文件: rm data/raw/hf_btc_hourly.*

如果模型加载仍然报错：
  → 删除旧模型并重新训练: rm models/saved/*.pth && python train.py

如果数据加载仍然报错：
  → 确保使用最新代码并删除所有缓存
  → 检查 traceback 中的文件路径是否指向正确的文件
""")

print("=" * 70)
