"""
SAM3 Demo Backend - Model Manager

This module provides the SAM3ModelManager class for managing SAM3 model
loading, initialization, and lifecycle. It implements a singleton pattern
to ensure efficient resource usage.

Based on the design document and SAM3 best practices:
- SAM3SemanticPredictor: For text prompts, bounding box prompts, image examples
- SAM model: For point prompts (SAM2 compatible mode)
"""
import os
import logging
from typing import Optional, Any

import torch

logger = logging.getLogger(__name__)


class SAM3ModelManager:
    """SAM3 模型管理器，单例模式
    
    负责 SAM3 模型的加载、初始化和生命周期管理。
    提供两种推理接口：
    - SAM3SemanticPredictor: 用于文本提示、边界框提示、图像范例分割
    - SAM 模型: 用于点击提示分割（SAM2 兼容模式）
    
    Attributes:
        model_path: 模型文件路径
        device: 推理设备 ("cuda" 或 "cpu")
        confidence: 默认置信度阈值
        use_half: 是否使用半精度推理
    """
    
    _instance: Optional["SAM3ModelManager"] = None
    
    def __new__(cls, *args, **kwargs) -> "SAM3ModelManager":
        """单例模式实现"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        confidence: float = 0.25,
        use_half: bool = True,
    ):
        """初始化模型管理器
        
        Args:
            model_path: 模型文件路径，默认从环境变量 SAM3_MODEL_PATH 读取，
                       或使用 "sam3.pt"（Ultralytics 会自动下载）
            device: 推理设备，默认自动检测 CUDA 可用性
            confidence: 置信度阈值，默认 0.25
            use_half: 是否使用半精度推理，默认 True（减少显存占用）
        """
        # 避免重复初始化
        if self._initialized:
            return

        # 配置模型路径
        self.model_path = model_path or os.environ.get("SAM3_MODEL_PATH", "sam3.pt")
        
        # 配置推理设备
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        self.confidence = confidence
        self.use_half = use_half and self.device == "cuda"  # 半精度仅在 GPU 上有效
        
        # 模型实例（延迟加载）
        self._semantic_predictor: Optional[Any] = None
        self._point_model: Optional[Any] = None
        
        self._initialized = True
        
        logger.info(
            f"SAM3ModelManager initialized: model_path={self.model_path}, "
            f"device={self.device}, half={self.use_half}"
        )
    
    def get_semantic_predictor(self) -> Any:
        """获取语义预测器实例
        
        用途：文本提示、边界框提示、图像范例分割
        特点：支持概念级分割，可检测所有匹配实例
        
        Returns:
            SAM3SemanticPredictor 实例
        """
        if self._semantic_predictor is None:
            logger.info("Loading SAM3SemanticPredictor...")
            
            from ultralytics.models.sam import SAM3SemanticPredictor
            
            overrides = dict(
                conf=self.confidence,
                task="segment",
                mode="predict",
                model=self.model_path,
                half=self.use_half,
                verbose=False,
            )
            self._semantic_predictor = SAM3SemanticPredictor(overrides=overrides)
            
            logger.info("SAM3SemanticPredictor loaded successfully")
        
        return self._semantic_predictor
    
    def get_point_model(self) -> Any:
        """获取点提示模型实例（SAM2 兼容模式）
        
        用途：点击提示分割
        特点：基于点击位置进行单实例分割
        
        Returns:
            SAM 模型实例
        """
        if self._point_model is None:
            logger.info("Loading SAM point model...")
            
            from ultralytics import SAM
            
            self._point_model = SAM(self.model_path)
            
            logger.info("SAM point model loaded successfully")
        
        return self._point_model
    
    def preload_models(self) -> None:
        """预加载所有模型到内存/GPU
        
        在应用启动时调用，确保首次请求时模型已就绪。
        """
        logger.info("Preloading SAM3 models...")
        
        # 预加载语义预测器
        self.get_semantic_predictor()
        
        # 预加载点击模型
        self.get_point_model()
        
        logger.info("All SAM3 models preloaded successfully")
    
    def is_ready(self) -> bool:
        """检查模型是否已加载就绪
        
        Returns:
            True 如果至少一个模型已加载
        """
        return self._semantic_predictor is not None or self._point_model is not None
    
    def is_fully_loaded(self) -> bool:
        """检查所有模型是否都已加载
        
        Returns:
            True 如果所有模型都已加载
        """
        return self._semantic_predictor is not None and self._point_model is not None
    
    def get_device_info(self) -> dict:
        """获取设备信息
        
        Returns:
            包含设备信息的字典
        """
        info = {
            "device": self.device,
            "cuda_available": torch.cuda.is_available(),
            "half_precision": self.use_half,
        }
        
        if torch.cuda.is_available():
            info["cuda_device_name"] = torch.cuda.get_device_name(0)
            info["cuda_memory_allocated_gb"] = round(
                torch.cuda.memory_allocated() / 1e9, 2
            )
            info["cuda_memory_reserved_gb"] = round(
                torch.cuda.memory_reserved() / 1e9, 2
            )
        
        return info
    
    def clear_cache(self) -> None:
        """清理 GPU 缓存
        
        在每次请求后调用，防止内存泄漏。
        """
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.debug("GPU cache cleared")
    
    @classmethod
    def reset_instance(cls) -> None:
        """重置单例实例（主要用于测试）"""
        if cls._instance is not None:
            cls._instance._semantic_predictor = None
            cls._instance._point_model = None
            cls._instance._initialized = False
            cls._instance = None


# 全局模型管理器实例
_model_manager: Optional[SAM3ModelManager] = None


def get_model_manager() -> SAM3ModelManager:
    """获取全局模型管理器实例
    
    Returns:
        SAM3ModelManager 单例实例
    """
    global _model_manager
    if _model_manager is None:
        _model_manager = SAM3ModelManager()
    return _model_manager


def initialize_model_manager(
    model_path: Optional[str] = None,
    device: Optional[str] = None,
    confidence: float = 0.25,
    use_half: bool = True,
    preload: bool = True,
) -> SAM3ModelManager:
    """初始化全局模型管理器
    
    Args:
        model_path: 模型文件路径
        device: 推理设备
        confidence: 置信度阈值
        use_half: 是否使用半精度
        preload: 是否预加载模型
    
    Returns:
        初始化后的 SAM3ModelManager 实例
    """
    global _model_manager
    
    # 重置现有实例
    SAM3ModelManager.reset_instance()
    
    # 创建新实例
    _model_manager = SAM3ModelManager(
        model_path=model_path,
        device=device,
        confidence=confidence,
        use_half=use_half,
    )
    
    # 预加载模型
    if preload:
        _model_manager.preload_models()
    
    return _model_manager
