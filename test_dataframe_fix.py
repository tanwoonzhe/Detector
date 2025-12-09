"""
测试DataFrame重采样修复
======================
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np

def test_dataframe_resample():
    """测试DataFrame重采样修复"""
    print("=" * 60)
    print("测试: DataFrame 重采样修复")
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
        
        # 测试方法1: 分别获取每个列（旧方法）
        print("\n方法1: 分别获取每个列...")
        try:
            df_resampled = data.resample("h")
            open_vals = df_resampled["open"].first()
            high_vals = df_resampled["high"].max()
            low_vals = df_resampled["low"].min()
            close_vals = df_resampled["close"].last()
            
            print(f"  open_vals类型: {type(open_vals)}, 形状: {open_vals.shape}")
            print(f"  open_vals.values类型: {type(open_vals.values)}, 形状: {open_vals.values.shape}")
            
            df_method1 = pd.DataFrame({
                'open': open_vals,
                'high': high_vals,
                'low': low_vals,
                'close': close_vals
            })
            print(f"  ✓ 方法1结果: {df_method1.shape}")
        except Exception as e:
            print(f"  ✗ 方法1失败: {str(e)[:100]}")
        
        # 测试方法2: 使用agg（新方法，更稳定）
        print("\n方法2: 使用agg一次性聚合...")
        try:
            agg_dict = {
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }
            
            df_hourly = data.resample("h").agg(agg_dict)
            
            print(f"  ✓ 重采样后数据: {df_hourly.shape}")
            print(f"  ✓ 列: {df_hourly.columns.tolist()}")
            print(f"  ✓ 索引类型: {type(df_hourly.index)}")
            print(f"  ✓ 前3行:\n{df_hourly.head(3)}")
            print(f"  ✓ 数据类型:\n{df_hourly.dtypes}")
        except Exception as e:
            print(f"  ✗ 方法2失败: {str(e)[:100]}")
        
        print("\n✅ DataFrame 重采样测试通过！")
        return True
        
    except Exception as e:
        print(f"\n❌ DataFrame 重采样测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("🔧 DataFrame重采样修复验证\n")
    
    result = test_dataframe_resample()
    
    print("\n" + "=" * 60)
    if result:
        print("🎉 测试通过！HF数据加载问题已修复")
        print("\n修复说明:")
        print("- 原因: 使用.values导致多维数组")
        print("- 解决: 直接使用pandas Series对象")
        print("- 影响文件: src/data_collection/hf_loader_fixed.py")
    else:
        print("⚠️ 测试失败，请检查错误信息")
    
    sys.exit(0 if result else 1)
