"""
模型管理器
================================
管理已训练模型的发现、加载和元数据
"""

import torch
import joblib
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class ModelInfo:
    """模型信息"""
    name: str                          # 模型名称 (如 gru_best)
    model_type: str                    # 模型类型 (GRU, BiLSTM, CNN-LSTM, LightGBM)
    file_path: Path                    # 文件路径
    file_size_mb: float               # 文件大小(MB)
    created_time: datetime            # 创建时间
    
    # 模型配置 (从checkpoint读取)
    input_shape: Optional[tuple] = None
    n_classes: int = 3
    hidden_size: Optional[int] = None
    num_layers: Optional[int] = None
    dropout: Optional[float] = None
    
    # 训练信息
    epochs_trained: Optional[int] = None
    best_val_accuracy: Optional[float] = None
    best_val_loss: Optional[float] = None
    
    # 额外信息
    extra_info: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'name': self.name,
            'model_type': self.model_type,
            'file_path': str(self.file_path),
            'file_size_mb': self.file_size_mb,
            'created_time': self.created_time.isoformat() if self.created_time else None,
            'input_shape': self.input_shape,
            'n_classes': self.n_classes,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'dropout': self.dropout,
            'epochs_trained': self.epochs_trained,
            'best_val_accuracy': self.best_val_accuracy,
            'best_val_loss': self.best_val_loss,
            'extra_info': self.extra_info
        }
    
    def get_summary(self) -> str:
        """获取摘要字符串"""
        parts = [f"类型: {self.model_type}"]
        if self.input_shape:
            parts.append(f"输入: {self.input_shape}")
        if self.hidden_size:
            parts.append(f"隐藏层: {self.hidden_size}")
        if self.best_val_accuracy:
            parts.append(f"验证准确率: {self.best_val_accuracy:.2%}")
        parts.append(f"大小: {self.file_size_mb:.1f}MB")
        return " | ".join(parts)


class ModelManager:
    """
    模型管理器
    
    功能:
    - 扫描已训练的模型
    - 读取模型元数据
    - 加载指定模型
    - 管理模型（删除、重命名等）
    """
    
    MODEL_EXTENSIONS = {
        '.pth': ['GRU', 'BiLSTM', 'CNN-LSTM'],  # PyTorch模型
        '.txt': ['LightGBM'],                   # LightGBM模型
        '.pkl': ['LightGBM'],                   # joblib保存的模型
    }
    
    def __init__(self, model_dir: Optional[Path] = None):
        """
        初始化
        
        Args:
            model_dir: 模型目录，默认为 models/saved/
        """
        if model_dir is None:
            model_dir = Path(__file__).parent.parent.parent / "models" / "saved"
        
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self._model_cache: Dict[str, Any] = {}
    
    def scan_models(self) -> List[ModelInfo]:
        """
        扫描所有已训练的模型
        
        Returns:
            模型信息列表
        """
        models = []
        
        for ext, model_types in self.MODEL_EXTENSIONS.items():
            for file_path in self.model_dir.glob(f"*{ext}"):
                try:
                    info = self._extract_model_info(file_path)
                    if info:
                        models.append(info)
                except Exception as e:
                    logger.warning(f"无法读取模型 {file_path}: {e}")
        
        # 按创建时间排序（最新的在前）
        models.sort(key=lambda x: x.created_time or datetime.min, reverse=True)
        
        return models
    
    def _extract_model_info(self, file_path: Path) -> Optional[ModelInfo]:
        """从模型文件提取信息"""
        ext = file_path.suffix.lower()
        
        # 基本信息
        stat = file_path.stat()
        file_size_mb = stat.st_size / (1024 * 1024)
        created_time = datetime.fromtimestamp(stat.st_mtime)
        
        # 根据文件名推断模型类型
        name = file_path.stem
        model_type = self._infer_model_type(name, ext)
        
        info = ModelInfo(
            name=name,
            model_type=model_type,
            file_path=file_path,
            file_size_mb=file_size_mb,
            created_time=created_time
        )
        
        # 尝试读取详细配置
        try:
            if ext == '.pth':
                self._read_pytorch_config(file_path, info)
            elif ext in ['.txt', '.pkl']:
                self._read_lightgbm_config(file_path, info)
        except Exception as e:
            logger.debug(f"无法读取 {file_path} 的详细配置: {e}")
        
        return info
    
    def _infer_model_type(self, name: str, ext: str) -> str:
        """从文件名推断模型类型"""
        name_lower = name.lower()
        
        if 'gru' in name_lower:
            return 'GRU'
        elif 'bilstm' in name_lower or 'bi_lstm' in name_lower:
            return 'BiLSTM'
        elif 'cnn' in name_lower and 'lstm' in name_lower:
            return 'CNN-LSTM'
        elif 'lightgbm' in name_lower or 'lgbm' in name_lower:
            return 'LightGBM'
        elif ext == '.pth':
            return 'PyTorch'
        elif ext in ['.txt', '.pkl']:
            return 'LightGBM'
        else:
            return 'Unknown'
    
    def _read_pytorch_config(self, file_path: Path, info: ModelInfo):
        """读取PyTorch模型配置"""
        checkpoint = torch.load(file_path, map_location='cpu')
        
        if 'config' in checkpoint:
            config = checkpoint['config']
            info.input_shape = tuple(config.get('input_shape', []))
            info.n_classes = config.get('n_classes', 3)
            info.hidden_size = config.get('hidden_size')
            info.num_layers = config.get('num_layers')
            info.dropout = config.get('dropout')
        
        if 'history' in checkpoint:
            history = checkpoint['history']
            if 'val_acc' in history and history['val_acc']:
                info.best_val_accuracy = max(history['val_acc'])
                info.epochs_trained = len(history['val_acc'])
            if 'val_loss' in history and history['val_loss']:
                info.best_val_loss = min(history['val_loss'])
    
    def _read_lightgbm_config(self, file_path: Path, info: ModelInfo):
        """读取LightGBM模型配置"""
        try:
            data = joblib.load(file_path)
            if isinstance(data, dict):
                if 'params' in data:
                    info.extra_info['params'] = data['params']
                if 'n_features_in' in data:
                    info.input_shape = (None, data['n_features_in'])
                if 'model' in data and hasattr(data['model'], 'best_score_'):
                    info.extra_info['best_score'] = data['model'].best_score_
        except:
            pass
    
    def get_model_info(self, name: str) -> Optional[ModelInfo]:
        """获取指定模型的信息"""
        models = self.scan_models()
        for model in models:
            if model.name == name:
                return model
        return None
    
    def load_model(self, name_or_path: Union[str, Path]):
        """
        加载模型
        
        Args:
            name_or_path: 模型名称或路径
            
        Returns:
            加载的模型实例
        """
        # 确定文件路径
        if isinstance(name_or_path, Path) or '/' in str(name_or_path) or '\\' in str(name_or_path):
            file_path = Path(name_or_path)
        else:
            # 按名称查找
            file_path = None
            for ext in self.MODEL_EXTENSIONS.keys():
                candidate = self.model_dir / f"{name_or_path}{ext}"
                if candidate.exists():
                    file_path = candidate
                    break
            
            if file_path is None:
                raise FileNotFoundError(f"找不到模型: {name_or_path}")
        
        # 检查缓存
        cache_key = str(file_path)
        if cache_key in self._model_cache:
            return self._model_cache[cache_key]
        
        # 加载模型
        model = self._load_model_file(file_path)
        self._model_cache[cache_key] = model
        
        return model
    
    def _load_model_file(self, file_path: Path):
        """根据文件类型加载模型"""
        ext = file_path.suffix.lower()
        model_type = self._infer_model_type(file_path.stem, ext)
        
        if ext == '.pth':
            return self._load_pytorch_model(file_path, model_type)
        elif ext in ['.txt', '.pkl']:
            return self._load_lightgbm_model(file_path)
        else:
            raise ValueError(f"不支持的模型格式: {ext}")
    
    def _load_pytorch_model(self, file_path: Path, model_type: str):
        """加载PyTorch模型"""
        from .gru import GRUPredictor
        from .bilstm import BiLSTMPredictor
        from .cnn_lstm import CNNLSTMPredictor
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if model_type == 'GRU':
            model = GRUPredictor(device=device)
        elif model_type == 'BiLSTM':
            model = BiLSTMPredictor(device=device)
        elif model_type == 'CNN-LSTM':
            model = CNNLSTMPredictor(device=device)
        else:
            # 尝试从checkpoint判断
            checkpoint = torch.load(file_path, map_location='cpu')
            if 'config' in checkpoint:
                # 根据配置推断
                model = GRUPredictor(device=device)  # 默认使用GRU
            else:
                raise ValueError(f"无法确定模型类型: {file_path}")
        
        model.load(file_path, auto_build=True)
        return model
    
    def _load_lightgbm_model(self, file_path: Path):
        """加载LightGBM模型"""
        from .lightgbm_model import LightGBMPredictor
        
        model = LightGBMPredictor()
        model.load(file_path)
        return model
    
    def delete_model(self, name: str) -> bool:
        """删除模型"""
        info = self.get_model_info(name)
        if info and info.file_path.exists():
            info.file_path.unlink()
            # 清除缓存
            cache_key = str(info.file_path)
            if cache_key in self._model_cache:
                del self._model_cache[cache_key]
            logger.info(f"已删除模型: {name}")
            return True
        return False
    
    def rename_model(self, old_name: str, new_name: str) -> bool:
        """重命名模型"""
        info = self.get_model_info(old_name)
        if info and info.file_path.exists():
            new_path = info.file_path.parent / f"{new_name}{info.file_path.suffix}"
            info.file_path.rename(new_path)
            logger.info(f"已重命名模型: {old_name} -> {new_name}")
            return True
        return False
    
    def get_models_by_type(self, model_type: str) -> List[ModelInfo]:
        """按类型获取模型"""
        return [m for m in self.scan_models() if m.model_type == model_type]
    
    def clear_cache(self):
        """清除模型缓存"""
        self._model_cache.clear()


# 全局模型管理器实例
model_manager = ModelManager()
