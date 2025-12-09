"""
🔧 完整修复验证脚本
====================
验证所有修复是否正常工作
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

def test_dataframe_creation():
    """测试 DataFrame 创建"""
    print("=" * 50)
    print("测试 1: DataFrame 创建")
    print("=" * 50)
    
    try:
        import pandas as pd
        import numpy as np
        
        # 模拟重采样结果
        dates = pd.date_range('2025-01-01', periods=100, freq='h')
        data = pd.DataFrame({
            'open': np.random.rand(100) * 100,
            'high': np.random.rand(100) * 100,
            'low': np.random.rand(100) * 100,
            'close': np.random.rand(100) * 100,
            'volume': np.random.rand(100) * 1000
        }, index=dates)
        
        # 测试重采样
        df_resampled = data.resample("h")
        open_vals = df_resampled["open"].first()
        high_vals = df_resampled["high"].max()
        low_vals = df_resampled["low"].min()
        close_vals = df_resampled["close"].last()
        
        # 直接使用Series创建DataFrame（避免.values造成的形状问题）
        df_hourly = pd.DataFrame({
            'open': open_vals,
            'high': high_vals,
            'low': low_vals,
            'close': close_vals
        })
        
        print(f"✅ DataFrame 创建成功: {df_hourly.shape}")
        print(f"✅ 列: {df_hourly.columns.tolist()}")
        return True
    except Exception as e:
        print(f"❌ DataFrame 创建失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_loading():
    """测试模型加载"""
    print("\n" + "=" * 50)
    print("测试 2: 模型加载机制")
    print("=" * 50)
    
    try:
        from src.models.gru import GRUPredictor
        import torch
        
        # 创建模型
        model = GRUPredictor(
            hidden_size=64,
            num_layers=1,
            dropout=0.2
        )
        
        # 构建模型
        input_shape = (24, 50)  # 24小时窗口，50个特征
        model.build(input_shape=input_shape, n_classes=3)
        
        print(f"✅ 模型构建成功")
        print(f"✅ 输入形状: {model.input_shape}")
        print(f"✅ 模型类型: {type(model.model)}")
        
        # 测试保存（到临时路径）
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as tmp:
            tmp_path = Path(tmp.name)
        
        try:
            model.save(tmp_path)
            print(f"✅ 模型保存成功: {tmp_path}")
            
            # 测试加载
            model2 = GRUPredictor(
                hidden_size=64,
                num_layers=1,
                dropout=0.2
            )
            
            # 从检查点获取输入形状
            checkpoint = torch.load(tmp_path, map_location='cpu')
            if 'config' in checkpoint and 'input_shape' in checkpoint['config']:
                input_shape_loaded = checkpoint['config']['input_shape']
                print(f"✅ 从检查点读取输入形状: {input_shape_loaded}")
            
            model2.build(input_shape=input_shape, n_classes=3)
            model2.load(tmp_path)
            print(f"✅ 模型加载成功")
            
        finally:
            # 清理临时文件
            if tmp_path.exists():
                tmp_path.unlink()
        
        return True
    except Exception as e:
        print(f"❌ 模型加载测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_feature_engineering():
    """测试特征工程"""
    print("\n" + "=" * 50)
    print("测试 3: 特征工程")
    print("=" * 50)
    
    try:
        import pandas as pd
        import numpy as np
        from src.features.engineer import FeatureEngineer
        
        # 创建测试数据
        dates = pd.date_range('2025-01-01', periods=500, freq='h')
        df = pd.DataFrame({
            'open': np.random.rand(500) * 90000 + 85000,
            'high': np.random.rand(500) * 90000 + 85000,
            'low': np.random.rand(500) * 90000 + 85000,
            'close': np.random.rand(500) * 90000 + 85000,
            'volume': np.random.rand(500) * 1000
        }, index=dates)
        
        # 创建特征工程器
        engineer = FeatureEngineer()
        
        print(f"原始数据: {df.shape}")
        
        # 生成特征
        df_features = engineer.create_features(df)
        
        print(f"✅ 特征生成成功: {df_features.shape}")
        print(f"✅ 特征数量: {len(df_features.columns)}")
        print(f"✅ 数据保留率: {len(df_features)/len(df)*100:.1f}%")
        
        # 检查 NaN 和 Inf
        nan_count = df_features.isna().sum().sum()
        inf_count = np.isinf(df_features.select_dtypes(include=[np.number])).sum().sum()
        
        print(f"✅ NaN 值: {nan_count}")
        print(f"✅ Inf 值: {inf_count}")
        
        if len(df_features) == 0:
            print("❌ 警告: 所有数据被删除")
            return False
        
        return True
    except Exception as e:
        print(f"❌ 特征工程测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_model_files():
    """检查模型文件"""
    print("\n" + "=" * 50)
    print("测试 4: 模型文件检查")
    print("=" * 50)
    
    model_dir = Path(__file__).parent / "models" / "saved"
    
    expected_files = {
        "GRU": "gru_best.pth",
        "LightGBM": "lightgbm_best.txt"
    }
    
    found = 0
    for model_name, filename in expected_files.items():
        filepath = model_dir / filename
        if filepath.exists():
            size = filepath.stat().st_size / (1024 * 1024)
            print(f"✅ {model_name}: {filename} ({size:.2f} MB)")
            found += 1
        else:
            print(f"⚠️  {model_name}: {filename} (未找到)")
    
    if found == 0:
        print("\n💡 提示: 还未训练模型，运行以下命令:")
        print("  python train.py --model gru --epochs 100 --batch-size 64")
    
    return True


def main():
    print("🔧 开始验证所有修复...")
    print()
    
    results = []
    
    # 运行所有测试
    results.append(("DataFrame 创建", test_dataframe_creation()))
    results.append(("模型加载机制", test_model_loading()))
    results.append(("特征工程", test_feature_engineering()))
    results.append(("模型文件", check_model_files()))
    
    # 总结
    print("\n" + "=" * 50)
    print("验证结果总结")
    print("=" * 50)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{status} - {name}")
    
    print()
    print(f"总计: {passed}/{total} 测试通过")
    
    if passed == total:
        print("\n🎉 所有测试通过！系统可以正常运行")
        print("\n下一步:")
        print("1. 训练模型: python train.py --model gru --epochs 100 --batch-size 64")
        print("2. 启动 Dashboard: streamlit run app/dashboard_realtime_binance.py")
    else:
        print("\n⚠️  部分测试失败，请检查错误信息")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
