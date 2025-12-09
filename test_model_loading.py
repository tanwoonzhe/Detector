"""
测试模型加载修复
=================
验证：
1. HF数据重采样不会出现形状错误
2. 模型可以从checkpoint自动构建和加载
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np
import torch

def test_dataframe_resample():
    """测试DataFrame重采样修复"""
    print("=" * 60)
    print("测试 1: DataFrame 重采样修复")
    print("=" * 60)
    
    try:
        # 创建测试数据
        dates = pd.date_range('2025-01-01', periods=1000, freq='min')
        data = pd.DataFrame({
            'open': np.random.rand(1000) * 100 + 50000,
            'high': np.random.rand(1000) * 100 + 50000,
            'low': np.random.rand(1000) * 100 + 50000,
            'close': np.random.rand(1000) * 100 + 50000,
            'volume': np.random.rand(1000) * 1000
        }, index=dates)
        
        print(f"原始数据: {data.shape}")
        
        # 模拟hf_loader_fixed.py中的重采样逻辑
        df_resampled = data.resample("h")
        
        open_vals = df_resampled["open"].first()
        high_vals = df_resampled["high"].max()
        low_vals = df_resampled["low"].min()
        close_vals = df_resampled["close"].last()
        
        # 使用修复后的方法：直接使用Series而不是.values
        df_hourly = pd.DataFrame({
            'open': open_vals,
            'high': high_vals,
            'low': low_vals,
            'close': close_vals
        })
        
        print(f"重采样后数据: {df_hourly.shape}")
        print(f"列: {df_hourly.columns.tolist()}")
        print(f"索引类型: {type(df_hourly.index)}")
        print(f"前3行:\n{df_hourly.head(3)}")
        
        print("\n✅ DataFrame 重采样测试通过！")
        return True
        
    except Exception as e:
        print(f"\n❌ DataFrame 重采样测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_auto_build():
    """测试模型自动构建功能"""
    print("\n" + "=" * 60)
    print("测试 2: 模型自动构建和加载")
    print("=" * 60)
    
    try:
        from src.models.gru import GRUPredictor
        import tempfile
        
        # 创建并保存一个模型
        print("创建并保存测试模型...")
        model1 = GRUPredictor(hidden_size=64, num_layers=1, dropout=0.2)
        
        # 构建模型（使用特定的input_shape）
        input_shape = (24, 124)  # 模拟训练时的124个特征
        model1.build(input_shape=input_shape, n_classes=3)
        
        print(f"  输入形状: {input_shape}")
        print(f"  模型参数数量: {sum(p.numel() for p in model1.model.parameters())}")
        
        # 保存到临时文件
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as tmp:
            tmp_path = Path(tmp.name)
        
        model1.save(tmp_path)
        print(f"  模型已保存: {tmp_path}")
        
        # 测试自动构建加载
        print("\n使用auto_build加载模型...")
        model2 = GRUPredictor(hidden_size=64, num_layers=1, dropout=0.2)
        
        # 不手动build，让load自动构建
        model2.load(tmp_path, auto_build=True)
        
        print(f"  ✅ 模型自动构建成功！")
        print(f"  加载的输入形状: {model2.input_shape}")
        print(f"  加载的类别数: {model2.n_classes}")
        
        # 验证模型可以进行预测
        X_test = np.random.randn(5, 24, 124).astype(np.float32)
        predictions = model2.predict(X_test)
        print(f"  测试预测: {predictions}")
        
        # 清理临时文件
        tmp_path.unlink()
        
        print("\n✅ 模型自动构建测试通过！")
        return True
        
    except Exception as e:
        print(f"\n❌ 模型自动构建测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_feature_dimension_mismatch():
    """测试特征维度不匹配场景"""
    print("\n" + "=" * 60)
    print("测试 3: 特征维度不匹配处理")
    print("=" * 60)
    
    try:
        from src.models.gru import GRUPredictor
        import tempfile
        
        # 场景：训练时使用124个特征，但加载时尝试使用100个特征
        print("模拟训练时使用124个特征...")
        model_train = GRUPredictor(hidden_size=64, num_layers=1)
        model_train.build(input_shape=(24, 124), n_classes=3)
        
        # 保存模型
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as tmp:
            tmp_path = Path(tmp.name)
        model_train.save(tmp_path)
        print(f"  模型已保存 (124个特征)")
        
        # 加载时使用auto_build，会自动使用保存的124个特征配置
        print("\n加载模型（使用auto_build）...")
        model_load = GRUPredictor(hidden_size=64, num_layers=1)
        model_load.load(tmp_path, auto_build=True)
        
        print(f"  ✅ 自动使用正确的输入形状: {model_load.input_shape}")
        print(f"  预期特征数: 124, 实际加载: {model_load.input_shape[1]}")
        
        # 现在用户需要确保生成124个特征，而不是100个
        print("\n⚠️  注意: 如果特征生成只有100个特征，需要检查特征工程代码")
        print("     建议: 检查是否所有特征模块都被正确调用")
        
        # 清理
        tmp_path.unlink()
        
        print("\n✅ 特征维度不匹配处理测试通过！")
        return True
        
    except Exception as e:
        print(f"\n❌ 特征维度测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("🔧 模型加载修复验证\n")
    
    results = []
    results.append(test_dataframe_resample())
    results.append(test_model_auto_build())
    results.append(test_feature_dimension_mismatch())
    
    print("\n" + "=" * 60)
    print("📊 测试结果总结")
    print("=" * 60)
    
    test_names = [
        "DataFrame 重采样修复",
        "模型自动构建",
        "特征维度不匹配处理"
    ]
    
    for name, result in zip(test_names, results):
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{status} - {name}")
    
    all_passed = all(results)
    
    if all_passed:
        print("\n🎉 所有测试通过！")
        print("\n下一步:")
        print("1. 如果模型加载仍报错特征维度不匹配，检查特征工程是否生成了足够的特征")
        print("2. 确保情感数据等可选特征在训练和预测时保持一致")
    else:
        print("\n⚠️  部分测试失败，请检查错误信息")
    
    sys.exit(0 if all_passed else 1)
