"""
测试 train.py 参数解析
"""
import argparse

parser = argparse.ArgumentParser(description='训练BTC趋势预测模型')
parser.add_argument('--model', type=str, default='gru',
                   choices=['gru', 'bilstm', 'cnn_lstm', 'lightgbm', 'all'],
                   help='要训练的模型')
parser.add_argument('--epochs', type=int, default=100,
                   help='训练轮数')
parser.add_argument('--batch_size', type=int, default=32,
                   help='批次大小')
parser.add_argument('--validate', action='store_true',
                   help='是否执行Walk-Forward验证')
parser.add_argument('--use-hf', action='store_true',
                   help='使用HuggingFace历史数据集')
parser.add_argument('--merge-recent', action='store_true',
                   help='合并最近的CoinGecko数据（与--use-hf一起使用）')

args = parser.parse_args()

print("✅ 参数解析成功!")
print(f"   模型: {args.model}")
print(f"   训练轮数: {args.epochs}")
print(f"   批次大小: {args.batch_size}")
print(f"   验证模式: {args.validate}")
print(f"   使用HF数据: {args.use_hf}")
print(f"   合并最新数据: {args.merge_recent}")
