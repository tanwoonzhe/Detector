"""
日志配置
================================
统一管理项目的日志输出，同时输出到控制台和文件
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional


def setup_logging(
    name: str = "detector",
    log_dir: Optional[Path] = None,
    level: int = logging.INFO,
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG
) -> logging.Logger:
    """
    配置日志系统
    
    Args:
        name: 日志器名称
        log_dir: 日志文件目录，默认为项目的 logs/ 目录
        level: 根日志级别
        console_level: 控制台输出级别
        file_level: 文件输出级别
        
    Returns:
        配置好的 logger
    """
    # 确定日志目录
    if log_dir is None:
        log_dir = Path(__file__).parent.parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建日志文件名（包含时间戳）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"{name}_{timestamp}.log"
    
    # 获取或创建 logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # 清除现有的 handlers（避免重复添加）
    logger.handlers.clear()
    
    # 日志格式
    detailed_format = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    simple_format = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # 控制台 Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(simple_format)
    logger.addHandler(console_handler)
    
    # 文件 Handler（详细日志）
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(file_level)
    file_handler.setFormatter(detailed_format)
    logger.addHandler(file_handler)
    
    # 同时创建一个最新日志的符号链接/副本
    latest_log = log_dir / f"{name}_latest.log"
    try:
        if latest_log.exists():
            latest_log.unlink()
        # Windows 上创建副本而非符号链接
        import shutil
        # 创建一个文件来追踪最新日志位置
        with open(latest_log, 'w', encoding='utf-8') as f:
            f.write(f"Latest log: {log_file}\n")
            f.write(f"Started at: {datetime.now().isoformat()}\n")
    except Exception:
        pass
    
    logger.info(f"日志文件: {log_file}")
    
    return logger


def get_logger(name: str = "detector") -> logging.Logger:
    """获取已配置的 logger，如果不存在则创建"""
    logger = logging.getLogger(name)
    if not logger.handlers:
        return setup_logging(name)
    return logger


# 为不同模块预配置的 logger 获取函数
def get_training_logger() -> logging.Logger:
    """获取训练日志器"""
    return setup_logging("training", level=logging.DEBUG)


def get_dashboard_logger() -> logging.Logger:
    """获取 Dashboard 日志器"""
    return setup_logging("dashboard", level=logging.INFO)


def get_data_logger() -> logging.Logger:
    """获取数据采集日志器"""
    return setup_logging("data", level=logging.INFO)


class TeeOutput:
    """
    同时输出到控制台和文件的输出流
    捕获所有 print() 语句的输出
    """
    def __init__(self, log_file: Path):
        self.terminal = sys.stdout
        self.log_file = open(log_file, 'a', encoding='utf-8')
        
    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()  # 确保立即写入
        
    def flush(self):
        self.terminal.flush()
        self.log_file.flush()
        
    def close(self):
        self.log_file.close()


class TeeError:
    """同时输出到控制台和文件的错误流"""
    def __init__(self, log_file: Path):
        self.terminal = sys.stderr
        self.log_file = open(log_file, 'a', encoding='utf-8')
        
    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(f"[STDERR] {message}")
        self.log_file.flush()
        
    def flush(self):
        self.terminal.flush()
        self.log_file.flush()
        
    def close(self):
        self.log_file.close()


def setup_full_logging(name: str = "app") -> Path:
    """
    设置完整日志，捕获所有 print() 和 stderr 输出
    
    Args:
        name: 日志名称前缀
        
    Returns:
        日志文件路径
    """
    log_dir = Path(__file__).parent.parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"{name}_{timestamp}.log"
    
    # 写入日志头
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write(f"=" * 60 + "\n")
        f.write(f"日志开始: {datetime.now().isoformat()}\n")
        f.write(f"程序: {name}\n")
        f.write(f"=" * 60 + "\n\n")
    
    # 重定向 stdout 和 stderr
    sys.stdout = TeeOutput(log_file)
    sys.stderr = TeeError(log_file)
    
    print(f"📝 所有输出将保存到: {log_file}")
    
    return log_file
