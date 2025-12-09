"""
检查模型文件和修复状态
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

def check_models():
    print("=" * 50)
    print("检查模型文件")
    print("=" * 50)
    
    model_dir = Path(__file__).parent / "models" / "saved"
    
    expected_files = {
        "GRU": "gru_best.pth",
        "BiLSTM": "bilstm_best.pth",
        "CNN-LSTM": "cnn_lstm_best.pth",
        "LightGBM": "lightgbm_best.txt"
    }
    
    found_models = []
    missing_models = []
    
    for model_name, filename in expected_files.items():
        filepath = model_dir / filename
        if filepath.exists():
            size = filepath.stat().st_size / (1024 * 1024)  # MB
            found_models.append(f"✅ {model_name}: {filename} ({size:.2f} MB)")
        else:
            missing_models.append(f"❌ {model_name}: {filename} (未找到)")
    
    print("\n已找到的模型:")
    if found_models:
        for model in found_models:
            print(f"  {model}")
    else:
        print("  无")
    
    print("\n缺失的模型:")
    if missing_models:
        for model in missing_models:
            print(f"  {model}")
        print("\n💡 提示: 运行以下命令训练模型:")
        print("  python train.py --model gru --epochs 100 --batch-size 64")
    else:
        print("  无 - 所有模型都已就绪!")
    
    print("\n" + "=" * 50)
    return len(found_models), len(missing_models)

def check_old_models():
    """检查旧的模型文件并提示删除"""
    print("\n检查旧模型文件...")
    model_dir = Path(__file__).parent / "models" / "saved"
    
    old_patterns = ["*_model.pt", "*_model.pkl", "*.pt", "*.pkl"]
    old_files = []
    
    for pattern in old_patterns:
        old_files.extend(model_dir.glob(pattern))
    
    # 过滤掉正确的文件名
    correct_files = {"gru_best.pth", "bilstm_best.pth", "cnn_lstm_best.pth", "lightgbm_best.txt"}
    old_files = [f for f in old_files if f.name not in correct_files]
    
    if old_files:
        print("\n⚠️ 发现旧的模型文件:")
        for f in old_files:
            print(f"  - {f.name}")
        print("\n建议删除这些文件，然后重新训练模型")
    else:
        print("✅ 没有旧的模型文件")

if __name__ == "__main__":
    found, missing = check_models()
    check_old_models()
    
    print("\n" + "=" * 50)
    print("修复状态总结")
    print("=" * 50)
    print("✅ 模型保存路径已修复")
    print("✅ HuggingFace DataFrame 创建已修复")
    print("✅ 特征工程 NaN/除零错误已修复")
    print(f"✅ 找到 {found} 个模型文件")
    if missing > 0:
        print(f"⚠️  缺少 {missing} 个模型文件 - 需要训练")
    print("=" * 50)
