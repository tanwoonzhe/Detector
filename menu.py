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
    
    # 调试输出
    print(f"\n📝 选择信息:")
    print(f"   数据源选择: {choice}")
    print(f"   模型选择: {model_choice} -> {model}")
    print(f"   训练轮数: {epochs}")
    
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
    print("1. 🚀 实时版 (Binance) - 秒级更新，真正的实时价格！")
    print("2. ⭐ 稳定版 (CoinGecko) - 小时级数据，适合训练")
    print("3. 完整版 (真实数据 + 所有侧边栏选项)")
    print("4. 简化版 (真实数据，快速测试)")
    print("5. 返回")
    
    choice = input("选择 (1-5): ").strip()
    
    if choice == "5":
        return
    
    import subprocess
    
    if choice == "1":
        cmd = "streamlit run app/dashboard_realtime_binance.py"
        print("\n🚀 实时版特点:")
        print("   ✓ Binance 公开 API (完全免费)")
        print("   ✓ 真正的实时价格 (秒级更新)")
        print("   ✓ 1分钟/5分钟/15分钟 K线")
        print("   ✓ 自动刷新 (5/10/15/30/60秒可选)")
        print("   ✓ RSI、布林带等技术指标")
    elif choice == "2":
        cmd = "streamlit run app/dashboard_stable.py"
        print("\n⭐ 稳定版特点:")
        print("   ✓ 使用缓存机制，避免无限重载")
        print("   ✓ 可选自动刷新 (15/30/60/120/300秒)")
        print("   ✓ 手动刷新按钮")
        print("   ✓ 智能交易信号计算")
    elif choice == "3":
        cmd = "streamlit run app/dashboard_complete.py"
    elif choice == "4":
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
            from src.data_collection.hf_loader_fixed import load_hf_btc_data
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
