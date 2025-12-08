"""
交互式启动菜单
================================
提供友好的命令行界面，选择训练或启动dashboard
"""

import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))


def print_banner():
    """打印欢迎横幅"""
    print("=" * 60)
    print("🚀 BTC趋势预测系统 v1.0")
    print("基于深度学习的加密货币趋势预测与交易信号生成")
    print("=" * 60)
    print()


def main_menu():
    """主菜单"""
    while True:
        print("\n📋 主菜单")
        print("-" * 40)
        print("1. 训练模型")
        print("2. 启动 Dashboard")
        print("3. 测试 CoinGecko API")
        print("4. 加载 HuggingFace 数据集")
        print("5. 退出")
        print("-" * 40)
        
        choice = input("请选择 (1-5): ").strip()
        
        if choice == "1":
            train_menu()
        elif choice == "2":
            launch_dashboard()
        elif choice == "3":
            test_coingecko()
        elif choice == "4":
            load_hf_dataset()
        elif choice == "5":
            print("\n👋 再见！")
            sys.exit(0)
        else:
            print("❌ 无效选项，请重新选择")


def train_menu():
    """训练子菜单"""
    print("\n🎓 模型训练")
    print("-" * 40)
    print("1. 使用 CoinGecko 实时数据")
    print("2. 使用 HuggingFace 历史数据集")
    print("3. 混合数据源（HF历史 + CoinGecko最新）")
    print("4. 返回主菜单")
    print("-" * 40)
    
    choice = input("选择数据源 (1-4): ").strip()
    
    if choice == "4":
        return
    
    # 模型选择
    print("\n选择模型:")
    print("1. GRU")
    print("2. BiLSTM")
    print("3. CNN-LSTM")
    print("4. LightGBM")
    print("5. 全部模型（集成）")
    
    model_choice = input("选择模型 (1-5): ").strip()
    model_map = {"1": "gru", "2": "bilstm", "3": "cnn_lstm", "4": "lightgbm", "5": "all"}
    model = model_map.get(model_choice, "gru")
    
    # 训练轮数
    epochs = input("训练轮数 (默认100): ").strip() or "100"
    
    # 构造命令
    import subprocess
    
    if choice == "1":
        cmd = f"python train.py --model {model} --epochs {epochs}"
    elif choice == "2":
        cmd = f"python train.py --model {model} --epochs {epochs} --use-hf"
    elif choice == "3":
        cmd = f"python train.py --model {model} --epochs {epochs} --use-hf --merge-recent"
    else:
        return
    
    print(f"\n▶️ 执行: {cmd}\n")
    subprocess.run(cmd, shell=True)
    input("\n按 Enter 继续...")


def launch_dashboard():
    """启动仪表板"""
    print("\n📊 启动 Dashboard...")
    print("-" * 40)
    print("1. 使用修正版 dashboard (显示真实价格)")
    print("2. 使用原版 dashboard")
    print("3. 返回")
    
    choice = input("选择 (1-3): ").strip()
    
    if choice == "3":
        return
    
    import subprocess
    
    if choice == "1":
        cmd = "streamlit run app/dashboard_fixed.py"
    else:
        cmd = "python main.py --dashboard"
    
    print(f"\n▶️ 执行: {cmd}")
    print("Dashboard 将在浏览器中打开...")
    print("按 Ctrl+C 停止\n")
    subprocess.run(cmd, shell=True)


def test_coingecko():
    """测试CoinGecko API"""
    print("\n🔍 测试 CoinGecko API...")
    import subprocess
    subprocess.run("python test_coingecko.py", shell=True)
    input("\n按 Enter 继续...")


def load_hf_dataset():
    """加载HuggingFace数据集"""
    print("\n📥 加载 HuggingFace 数据集...")
    print("这将下载并缓存 WinkingFace/CryptoLM-Bitcoin-BTC-USDT 数据集")
    confirm = input("继续? (y/n): ").strip().lower()
    
    if confirm == "y":
        try:
            from src.data_collection.hf_loader import load_hf_btc_data
            df = load_hf_btc_data()
            if not df.empty:
                print(f"\n✅ 成功加载 {len(df)} 条记录")
                print(f"时间范围: {df.index.min()} ~ {df.index.max()}")
                print(df.head())
            else:
                print("\n❌ 加载失败")
        except Exception as e:
            print(f"\n❌ 错误: {e}")
    
    input("\n按 Enter 继续...")


if __name__ == "__main__":
    try:
        print_banner()
        main_menu()
    except KeyboardInterrupt:
        print("\n\n👋 已取消")
        sys.exit(0)
