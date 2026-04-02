"""
SAM3 Demo Backend - Segmentation Service

This module provides the SegmentationService class that encapsulates
all SAM3 inference logic for different prompt types.

Based on the design document:
- Text prompts: Use SAM3SemanticPredictor
- Point prompts: Use SAM model (SAM2 compatible mode)
- Box prompts: Use SAM3SemanticPredictor
"""
import base64
import io
import logging
import time
import uuid
from typing import List, Tuple, Optional, Dict

import cv2
import numpy as np
from PIL import Image

from ..models.schemas import (
    MaskData,
    SegmentationResult,
    DefectBox,
    CachedSample,
    SampleInferResult,
    MaskDataWithCategory,
    BatchSampleInferResult,
    SampleListItem,
)
from .model_manager import SAM3ModelManager, get_model_manager
from .exceptions import SampleNotFoundError

logger = logging.getLogger(__name__)


def stitch_images(
    sample: np.ndarray,
    target: np.ndarray,
) -> tuple:
    """水平拼接样品图（左）和目标图（右）。

    高度不同时，较矮的图像底部填充黑色像素。

    Args:
        sample: 样品图像 (H1, W1, 3) RGB 格式
        target: 目标图像 (H2, W2, 3) RGB 格式

    Returns:
        tuple: (stitched, sample_width, target_width)
            - stitched: 拼接后图像 (max(H1,H2), W1+W2, 3)
            - sample_width: W1
            - target_width: W2
    """
    h1, w1 = sample.shape[:2]
    h2, w2 = target.shape[:2]
    max_h = max(h1, h2)

    # 创建黑色画布
    stitched = np.zeros((max_h, w1 + w2, 3), dtype=sample.dtype)

    # 放置样品图（左）
    stitched[:h1, :w1] = sample
    # 放置目标图（右）
    stitched[:h2, w1:w1 + w2] = target

    return stitched, w1, w2


def filter_and_offset_results(
    masks: np.ndarray,
    boxes: np.ndarray,
    scores: np.ndarray,
    sample_width: int,
    target_width: int,
    target_height: int,
    stitched_height: int,
    stitched_width: int,
) -> tuple:
    """过滤样品区域结果，偏移目标区域坐标，裁剪掩码。

    过滤规则：bbox 中心点在目标区域内（center_x > sample_width）的结果被保留。
    偏移规则：保留结果的 bbox x 坐标减去 sample_width，并裁剪到有效范围。
    裁剪规则：掩码先缩放到拼接图尺寸，然后从 x=sample_width 开始裁剪，
              宽度为 target_width，高度裁剪为 target_height。

    Args:
        masks: (N, H, W) 布尔掩码数组（可能是缩放后的尺寸）
        boxes: (N, 4) 或 (N, 6) 边界框数组 [x1, y1, x2, y2, ...]（原始图像坐标）
        scores: (N,) 置信度数组
        sample_width: W1（样品图宽度）
        target_width: W2（目标图宽度）
        target_height: H2（目标图高度）
        stitched_height: 拼接图高度
        stitched_width: 拼接图宽度

    Returns:
        tuple: (filtered_masks, filtered_boxes, filtered_scores)
            - filtered_masks: (M, target_height, target_width)
            - filtered_boxes: (M, 4) 偏移后的坐标（已裁剪到有效范围）
            - filtered_scores: (M,)
    """
    if len(masks) == 0:
        return (
            np.empty((0, target_height, target_width), dtype=masks.dtype),
            np.empty((0, 4), dtype=boxes.dtype),
            np.empty((0,), dtype=scores.dtype),
        )

    # 过滤：保留中心点在目标区域内的结果
    # 这比只检查 x2 > sample_width 更合理，避免保留主体在样品区域的对象
    center_x = (boxes[:, 0] + boxes[:, 2]) / 2
    keep = center_x > sample_width

    filtered_masks = masks[keep]
    filtered_boxes = boxes[keep][:, :4].copy()
    filtered_scores = scores[keep]

    if len(filtered_boxes) == 0:
        return (
            np.empty((0, target_height, target_width), dtype=masks.dtype),
            np.empty((0, 4), dtype=boxes.dtype),
            np.empty((0,), dtype=scores.dtype),
        )

    # 偏移：x 坐标减去 sample_width，并裁剪到有效范围 [0, target_width]
    filtered_boxes[:, 0] = np.clip(filtered_boxes[:, 0] - sample_width, 0, target_width)
    filtered_boxes[:, 2] = np.clip(filtered_boxes[:, 2] - sample_width, 0, target_width)
    # y 坐标裁剪到有效范围 [0, target_height]
    filtered_boxes[:, 1] = np.clip(filtered_boxes[:, 1], 0, target_height)
    filtered_boxes[:, 3] = np.clip(filtered_boxes[:, 3], 0, target_height)

    # 将掩码缩放到拼接图尺寸，然后裁剪目标区域
    result_masks = []
    for mask in filtered_masks:
        # 掩码可能是缩放后的尺寸，需要先缩放到拼接图尺寸
        mask_h, mask_w = mask.shape
        if mask_h != stitched_height or mask_w != stitched_width:
            # 使用 OpenCV 缩放掩码
            mask_resized = cv2.resize(
                mask.astype(np.uint8),
                (stitched_width, stitched_height),
                interpolation=cv2.INTER_NEAREST
            ).astype(mask.dtype)
        else:
            mask_resized = mask

        # 裁剪目标区域
        cropped = mask_resized[:target_height, sample_width:sample_width + target_width]
        result_masks.append(cropped)

    if result_masks:
        filtered_masks = np.stack(result_masks, axis=0)
    else:
        filtered_masks = np.empty((0, target_height, target_width), dtype=masks.dtype)

    return filtered_masks, filtered_boxes, filtered_scores


class SegmentationService:
    """分割服务，封装 SAM3 推理逻辑
    
    提供三种分割方式：
    - segment_with_text: 文本提示分割
    - segment_with_points: 点击提示分割
    - segment_with_boxes: 边界框提示分割
    
    Attributes:
        model_manager: SAM3 模型管理器实例
    """
    
    def __init__(self, model_manager: Optional[SAM3ModelManager] = None):
        """初始化分割服务
        
        Args:
            model_manager: 模型管理器实例，默认使用全局实例
        """
        self.model_manager = model_manager or get_model_manager()
        # 样本缓存字典：sample_id -> CachedSample
        self._sample_cache: Dict[str, "CachedSample"] = {}
    
    async def create_sample(
        self,
        image: np.ndarray,
        boxes: List[DefectBox],
    ) -> Tuple[str, float]:
        """创建缺陷样本并缓存样品图像
        
        缓存样品图像和边界框信息，用于后续拼接推理。
        
        Args:
            image: 样品图像 (H, W, C) RGB 格式
            boxes: 缺陷边界框列表
            
        Returns:
            Tuple[str, float]: (sample_id, cache_time_ms)
                - sample_id: 样本唯一标识
                - cache_time_ms: 缓存耗时（毫秒）
        """
        start_time = time.time()
        sample_id = str(uuid.uuid4())
        
        logger.info(f"Creating sample {sample_id} with {len(boxes)} boxes")
        
        # 缓存样品图像和边界框
        self._sample_cache[sample_id] = CachedSample(
            sample_image=image.copy(),  # 复制图像避免外部修改
            src_shape=image.shape[:2],  # (height, width)
            boxes=boxes,
        )
        
        cache_time_ms = (time.time() - start_time) * 1000
        
        logger.info(
            f"Sample {sample_id} created: {len(boxes)} boxes, "
            f"image shape {image.shape}, {cache_time_ms:.2f}ms"
        )
        
        return sample_id, cache_time_ms
    
    def delete_sample(self, sample_id: str) -> bool:
        """删除样本并释放缓存
        
        Args:
            sample_id: 样本唯一标识
            
        Returns:
            bool: 是否成功删除（True 表示存在并已删除，False 表示不存在）
        """
        if sample_id in self._sample_cache:
            del self._sample_cache[sample_id]
            logger.info(f"Sample {sample_id} deleted")
            return True
        
        logger.warning(f"Sample {sample_id} not found for deletion")
        return False
    
    def list_samples(self) -> List[SampleListItem]:
        """列出当前缓存的所有样本
        
        Returns:
            List[SampleListItem]: 样本列表
        """
        samples = []
        for sample_id, cached in self._sample_cache.items():
            samples.append(SampleListItem(
                sample_id=sample_id,
                boxes_count=len(cached.boxes),
                created_at=cached.created_at,
            ))
        
        logger.info(f"Listed {len(samples)} samples")
        return samples
    
    def _compute_iou(self, box1: List[float], box2: List[float]) -> float:
        """计算两个边界框的 IoU（交并比）
        
        Args:
            box1: [x1, y1, x2, y2]
            box2: [x1, y1, x2, y2]
            
        Returns:
            float: IoU 值 (0-1)
        """
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _match_mask_to_box(
        self, 
        mask_bbox: List[float], 
        input_boxes: List[DefectBox],
        sample_width: int,
    ) -> Optional[int]:
        """根据 IoU 匹配 mask 到输入 bbox
        
        Args:
            mask_bbox: mask 的边界框 [x1, y1, x2, y2]（已偏移到目标图坐标）
            input_boxes: 输入的缺陷边界框列表（样品图坐标）
            sample_width: 样品图宽度（用于坐标转换）
            
        Returns:
            Optional[int]: 匹配的 bbox 索引，如果没有匹配则返回 None
        """
        best_iou = 0.0
        best_idx = None
        
        for idx, box in enumerate(input_boxes):
            # 输入 bbox 是样品图坐标，需要转换到目标图坐标系
            # 由于样品图和目标图是拼接的，样品图的 bbox 在拼接图中的位置不变
            # 但我们比较的是目标图区域的 mask，所以直接比较相对位置
            input_bbox = box.to_list()
            iou = self._compute_iou(mask_bbox, input_bbox)
            if iou > best_iou:
                best_iou = iou
                best_idx = idx
        
        # 阈值防止误匹配
        return best_idx if best_iou > 0.1 else None
    
    async def infer_with_sample(
        self,
        sample_id: str,
        target_image: np.ndarray,
        confidence: float = 0.25,
    ) -> SampleInferResult:
        """使用缓存样品图对单张目标图进行拼接推理
        
        将缓存的样品图与目标图拼接，使用边界框进行分割推理。
        
        Args:
            sample_id: 样本唯一标识
            target_image: 目标图像 (H, W, C) RGB 格式
            confidence: 置信度阈值
            
        Returns:
            SampleInferResult: 推理结果（包含类别信息）
            
        Raises:
            SampleNotFoundError: 样本不存在
        """
        start_time = time.time()
        
        cached = self._sample_cache.get(sample_id)
        if not cached:
            raise SampleNotFoundError(sample_id)
        
        logger.info(f"Inferring with sample {sample_id} on target image")
        
        try:
            predictor = self.model_manager.get_semantic_predictor()
            
            # 临时设置置信度阈值
            original_conf = predictor.args.conf
            predictor.args.conf = confidence
            
            try:
                # 拼接样品图和目标图
                stitched_image, sample_width, target_width = stitch_images(
                    cached.sample_image, target_image
                )
                target_h, target_w = target_image.shape[:2]
                stitched_h, stitched_w = stitched_image.shape[:2]
                
                logger.info(
                    f"Stitched image: {stitched_h}x{stitched_w}, "
                    f"sample_width={sample_width}, target_width={target_width}"
                )
                
                # 设置拼接图像
                predictor.set_image(stitched_image)
                
                # 使用缓存的边界框进行推理（边界框在样品图区域）
                bboxes = [box.to_list() for box in cached.boxes]
                results = predictor(bboxes=bboxes)
                
                # 处理结果
                masks_data: List[MaskDataWithCategory] = []
                
                for r in results:
                    if r.masks is not None and r.boxes is not None:
                        # 获取原始数据
                        masks_np = r.masks.data.cpu().numpy()
                        boxes_np = r.boxes.xyxy.cpu().numpy()
                        scores_np = r.boxes.conf.cpu().numpy()
                        
                        # 过滤和偏移结果（只保留目标区域的检测）
                        filtered_masks, filtered_boxes, filtered_scores = filter_and_offset_results(
                            masks_np,
                            boxes_np,
                            scores_np,
                            sample_width,
                            target_width,
                            target_h,
                            stitched_h,
                            stitched_w,
                        )
                        
                        logger.info(
                            f"Filtered {len(masks_np)} -> {len(filtered_masks)} masks"
                        )
                        
                        # 处理每个过滤后的 mask
                        for i in range(len(filtered_masks)):
                            mask_np = (filtered_masks[i] * 255).astype(np.uint8)
                            bbox = filtered_boxes[i].tolist()
                            score = float(filtered_scores[i])
                            
                            # 编码为 Base64 PNG
                            mask_base64 = self._encode_mask_to_base64(mask_np)
                            
                            # 计算掩码面积
                            area = int(np.sum(mask_np > 0))
                            
                            # 使用 IoU 匹配确定类别
                            matched_idx = self._match_mask_to_box(
                                bbox, cached.boxes, sample_width
                            )
                            category = None
                            if matched_idx is not None:
                                category = cached.boxes[matched_idx].category
                            
                            masks_data.append(MaskDataWithCategory(
                                mask_base64=mask_base64,
                                bbox=bbox,
                                score=score,
                                label=category,
                                area=area,
                                category=category,
                            ))
                
                processing_time_ms = (time.time() - start_time) * 1000
                
                logger.info(
                    f"Sample inference completed: {len(masks_data)} masks, "
                    f"{processing_time_ms:.2f}ms"
                )
                
                return SampleInferResult(
                    masks=masks_data,
                    count=len(masks_data),
                    processing_time_ms=processing_time_ms,
                    image_size=(target_w, target_h),
                )
                
            finally:
                # 恢复原始置信度
                predictor.args.conf = original_conf
                
        finally:
            self.model_manager.clear_cache()
    
    async def batch_infer_with_sample(
        self,
        sample_id: str,
        target_images: List[np.ndarray],
        confidence: float = 0.25,
    ) -> BatchSampleInferResult:
        """使用缓存样品图批量处理多张目标图
        
        将缓存的样品图与每张目标图拼接进行批量分割推理。
        单张图失败时继续处理其他图片。
        
        Args:
            sample_id: 样本唯一标识
            target_images: 目标图像列表
            confidence: 置信度阈值
            
        Returns:
            BatchSampleInferResult: 批量推理结果（包含性能统计）
            
        Raises:
            SampleNotFoundError: 样本不存在
        """
        batch_start_time = time.time()
        
        cached = self._sample_cache.get(sample_id)
        if not cached:
            raise SampleNotFoundError(sample_id)
        
        logger.info(
            f"Batch inferring with sample {sample_id} on {len(target_images)} images"
        )
        
        results: List[SampleInferResult] = []
        success_count = 0
        failed_count = 0
        total_inference_time_ms = 0.0
        
        # 样品图缓存节省的是重复上传和解码的时间
        # 每次上传和解码约需 50ms
        estimated_upload_time_ms = 50.0
        sample_cache_saved_ms = estimated_upload_time_ms * len(target_images)
        
        for idx, target_image in enumerate(target_images):
            try:
                result = await self.infer_with_sample(
                    sample_id=sample_id,
                    target_image=target_image,
                    confidence=confidence,
                )
                results.append(result)
                success_count += 1
                total_inference_time_ms += result.processing_time_ms
                
            except Exception as e:
                logger.error(
                    f"Batch inference failed for image {idx}: {e}"
                )
                failed_count += 1
                
                # 返回空结果
                height, width = target_image.shape[:2]
                results.append(SampleInferResult(
                    masks=[],
                    count=0,
                    processing_time_ms=0.0,
                    image_size=(width, height),
                ))
        
        total_time_ms = (time.time() - batch_start_time) * 1000
        
        logger.info(
            f"Batch inference completed: {success_count}/{len(target_images)} success, "
            f"{total_time_ms:.2f}ms total, "
            f"~{sample_cache_saved_ms:.2f}ms saved by sample caching"
        )
        
        return BatchSampleInferResult(
            results=results,
            total=len(target_images),
            success_count=success_count,
            failed_count=failed_count,
            total_time_ms=total_time_ms,
            feature_reuse_saved_ms=sample_cache_saved_ms,
        )
    
    async def segment_with_text(
        self,
        image: np.ndarray,
        prompts: List[str],
        confidence: float = 0.25,
    ) -> SegmentationResult:
        """文本提示分割 - 使用 SAM3SemanticPredictor
        
        推理流程：
        1. 获取 SAM3SemanticPredictor 实例
        2. 调用 set_image() 提取图像特征
        3. 调用 predictor(text=prompts) 执行文本提示推理
        4. 返回所有匹配概念的掩码、边界框和置信度
        
        Args:
            image: 输入图像 (H, W, C) RGB 格式
            prompts: 文本提示列表，如 ["person", "car"]
            confidence: 置信度阈值
            
        Returns:
            SegmentationResult: 分割结果
        """
        start_time = time.time()
        
        logger.info(f"Starting text segmentation with prompts: {prompts}")

        try:
            predictor = self.model_manager.get_semantic_predictor()
            
            # 设置图像（提取特征）
            predictor.set_image(image)
            
            # 执行文本提示推理
            results = predictor(text=prompts)
            
            # 处理结果
            return self._process_semantic_results(
                results=results,
                image=image,
                start_time=start_time,
                labels=prompts,
            )
        
        finally:
            # 清理 GPU 缓存
            self.model_manager.clear_cache()
    
    async def segment_with_points(
        self,
        image: np.ndarray,
        points: List[Tuple[float, float]],
        labels: List[int],
    ) -> SegmentationResult:
        """点击提示分割 - 使用 SAM 模型（SAM2 兼容模式）
        
        推理流程：
        1. 获取 SAM 模型实例
        2. 调用 model.predict() 传入点坐标和标签
        3. 返回包含前景点、排除背景点的单个掩码
        
        注意：点击分割使用 SAM 模型而非 SAM3SemanticPredictor，
        因为点击是实例级操作，不需要概念理解能力。
        
        Args:
            image: 输入图像 (H, W, C) RGB 格式
            points: 点击坐标列表 [(x, y), ...]
            labels: 点标签列表 (1=前景, 0=背景)
            
        Returns:
            SegmentationResult: 分割结果
        """
        start_time = time.time()
        
        logger.info(f"Starting point segmentation with {len(points)} points")
        
        try:
            model = self.model_manager.get_point_model()
            
            # 执行点击提示推理
            results = model.predict(
                source=image,
                points=points,
                labels=labels,
            )
            
            # 处理结果
            return self._process_point_results(
                results=results,
                image=image,
                start_time=start_time,
            )
        
        finally:
            # 清理 GPU 缓存
            self.model_manager.clear_cache()
    
    async def segment_with_boxes(
        self,
        image: np.ndarray,
        boxes: List[List[float]],
    ) -> SegmentationResult:
        """边界框提示分割 - 使用 SAM3SemanticPredictor
        
        推理流程：
        1. 获取 SAM3SemanticPredictor 实例
        2. 调用 set_image() 提取图像特征
        3. 调用 predictor(bboxes=boxes) 执行边界框提示推理
        4. 返回边界框区域内及所有相似实例的掩码
        
        Args:
            image: 输入图像 (H, W, C) RGB 格式
            boxes: 边界框列表 [[x1, y1, x2, y2], ...]
            
        Returns:
            SegmentationResult: 分割结果
        """
        start_time = time.time()
        
        logger.info(f"Starting box segmentation with {len(boxes)} boxes")
        
        try:
            predictor = self.model_manager.get_semantic_predictor()
            
            # 设置图像（提取特征）
            predictor.set_image(image)
            
            # 执行边界框提示推理
            results = predictor(bboxes=boxes)
            
            # 处理结果
            return self._process_semantic_results(
                results=results,
                image=image,
                start_time=start_time,
            )
        
        finally:
            # 清理 GPU 缓存
            self.model_manager.clear_cache()
    

    # Maximum stitched image dimension (width or height)
    MAX_STITCH_DIMENSION = 8192

    async def segment_with_stitch(
        self,
        target_image: np.ndarray,
        sample_image: np.ndarray,
        sample_box: List[float],
        confidence: float = 0.25,
    ) -> SegmentationResult:
        """拼接分割 - 将样品图和目标图拼接后使用 BBox 推理

        推理流程：
        1. 调用 stitch_images 水平拼接样品图（左）和目标图（右）
        2. 检查拼接图尺寸是否超限（宽或高 > 8192px）
        3. 在拼接图上 set_image() 提取特征
        4. 使用原始 sample_box 坐标调用 predictor(bboxes=...) 推理
        5. 调用 filter_and_offset_results 过滤样品区域结果、偏移坐标、裁剪掩码

        Args:
            target_image: 目标图像 (H2, W2, 3) RGB 格式
            sample_image: 样品图像 (H1, W1, 3) RGB 格式
            sample_box: 样品边界框 [x1, y1, x2, y2]
            confidence: 置信度阈值 (0-1)，默认 0.25

        Returns:
            SegmentationResult: 分割结果（坐标已还原为目标图坐标系）

        Raises:
            ValueError: 拼接图尺寸超出限制
        """
        start_time = time.time()

        logger.info(f"Starting stitch segmentation with sample_box: {sample_box}, confidence: {confidence}")

        try:
            # 1. 拼接图像
            stitched, sample_width, target_width = stitch_images(sample_image, target_image)
            stitch_h, stitch_w = stitched.shape[:2]

            # 2. 尺寸超限检查
            if stitch_w > self.MAX_STITCH_DIMENSION or stitch_h > self.MAX_STITCH_DIMENSION:
                raise ValueError(
                    f"拼接后图像尺寸超出限制: {stitch_w}x{stitch_h}, "
                    f"最大允许 {self.MAX_STITCH_DIMENSION}px"
                )

            # 3. 在拼接图上推理
            predictor = self.model_manager.get_semantic_predictor()
            # 临时设置置信度阈值
            original_conf = predictor.args.conf
            predictor.args.conf = confidence
            try:
                predictor.set_image(stitched)
                results = predictor(bboxes=[sample_box])
            finally:
                # 恢复原始置信度
                predictor.args.conf = original_conf

            # 4. 提取原始掩码和边界框
            masks_list = []
            boxes_list = []
            scores_list = []

            for r in results:
                if r.masks is not None and len(r.masks.data) > 0:
                    masks_np = r.masks.data.cpu().numpy()
                    boxes_np = r.boxes.xyxy.cpu().numpy()
                    scores_np = r.boxes.conf.cpu().numpy()
                    logger.info(f"Raw masks shape: {masks_np.shape}, boxes shape: {boxes_np.shape}")
                    masks_list.append(masks_np)
                    boxes_list.append(boxes_np)
                    scores_list.append(scores_np)

            if masks_list:
                all_masks = np.concatenate(masks_list, axis=0)
                all_boxes = np.concatenate(boxes_list, axis=0)
                all_scores = np.concatenate(scores_list, axis=0)
            else:
                # No detections
                target_h = target_image.shape[0]
                processing_time_ms = (time.time() - start_time) * 1000
                return SegmentationResult(
                    masks=[],
                    count=0,
                    processing_time_ms=processing_time_ms,
                    image_size=(target_width, target_h),
                )

            # 5. 过滤 + 偏移 + 裁剪
            target_h = target_image.shape[0]
            logger.info(f"Stitched image shape: {stitched.shape}, sample_width: {sample_width}, target_width: {target_width}, target_h: {target_h}")
            logger.info(f"All masks shape before filter: {all_masks.shape}")
            filtered_masks, filtered_boxes, filtered_scores = filter_and_offset_results(
                all_masks, all_boxes, all_scores,
                sample_width, target_width, target_h,
                stitch_h, stitch_w,
            )
            logger.info(f"Filtered masks shape: {filtered_masks.shape}")

            # 6. 构建结果
            masks_data: List[MaskData] = []
            for i in range(len(filtered_masks)):
                mask_np = filtered_masks[i].astype(np.uint8) * 255
                
                # 调试：检查掩码内容与 bbox 的一致性
                rows = np.any(mask_np > 0, axis=1)
                cols = np.any(mask_np > 0, axis=0)
                if np.any(rows) and np.any(cols):
                    y_indices = np.where(rows)[0]
                    x_indices = np.where(cols)[0]
                    mask_y1, mask_y2 = y_indices[0], y_indices[-1]
                    mask_x1, mask_x2 = x_indices[0], x_indices[-1]
                    bbox_debug = filtered_boxes[i]
                    logger.info(
                        f"Mask {i}: shape={mask_np.shape}, "
                        f"mask_region=({mask_x1},{mask_y1})-({mask_x2},{mask_y2}), "
                        f"bbox=({bbox_debug[0]:.0f},{bbox_debug[1]:.0f})-({bbox_debug[2]:.0f},{bbox_debug[3]:.0f})"
                    )
                
                mask_base64 = self._encode_mask_to_base64(mask_np)
                bbox = filtered_boxes[i].tolist()
                score = float(filtered_scores[i])
                area = int(np.sum(mask_np > 0))

                masks_data.append(MaskData(
                    mask_base64=mask_base64,
                    bbox=bbox,
                    score=score,
                    label=None,
                    area=area,
                ))

            processing_time_ms = (time.time() - start_time) * 1000

            logger.info(
                f"Stitch segmentation completed: {len(masks_data)} masks, "
                f"{processing_time_ms:.2f}ms"
            )

            return SegmentationResult(
                masks=masks_data,
                count=len(masks_data),
                processing_time_ms=processing_time_ms,
                image_size=(target_width, target_h),
            )

        finally:
            self.model_manager.clear_cache()


    async def batch_segment_with_text(
        self,
        images: List[np.ndarray],
        prompts: List[str],
        confidence: float = 0.25,
    ) -> List[SegmentationResult]:
        """批量文本提示分割

        逐张图像串行处理，每张图像使用相同的文本提示。

        Args:
            images: 输入图像列表
            prompts: 文本提示列表（所有图像使用相同提示）
            confidence: 置信度阈值

        Returns:
            List[SegmentationResult]: 分割结果列表
        """
        results = []
        logger.info(f"Starting batch text segmentation: {len(images)} images, prompts: {prompts}")

        try:
            predictor = self.model_manager.get_semantic_predictor()

            for idx, image in enumerate(images):
                start_time = time.time()
                try:
                    predictor.set_image(image)
                    result = predictor(text=prompts)
                    results.append(self._process_semantic_results(
                        result, image, start_time, labels=prompts,
                    ))
                except Exception as e:
                    logger.error(f"Batch text segmentation failed for image {idx}: {e}")
                    # Return empty result for failed images
                    height, width = image.shape[:2]
                    results.append(SegmentationResult(
                        masks=[],
                        count=0,
                        processing_time_ms=(time.time() - start_time) * 1000,
                        image_size=(width, height),
                    ))
        finally:
            self.model_manager.clear_cache()

        return results


    async def batch_segment_with_stitch(
        self,
        target_images: List[np.ndarray],
        sample_image: np.ndarray,
        sample_box: List[float],
        confidence: float = 0.25,
    ) -> List[SegmentationResult]:
        """批量拼接分割，逐张目标图处理

        对每张目标图调用 segment_with_stitch。单张失败时返回空结果，
        继续处理其余图片。

        Args:
            target_images: 目标图像列表
            sample_image: 样品图像
            sample_box: 样品边界框 [x1, y1, x2, y2]
            confidence: 置信度阈值 (0-1)，默认 0.25

        Returns:
            List[SegmentationResult]: 分割结果列表，长度与 target_images 相同
        """
        results = []
        logger.info(f"Starting batch stitch segmentation: {len(target_images)} images, confidence: {confidence}")

        for idx, target_image in enumerate(target_images):
            try:
                result = await self.segment_with_stitch(
                    target_image=target_image,
                    sample_image=sample_image,
                    sample_box=sample_box,
                    confidence=confidence,
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Batch stitch segmentation failed for image {idx}: {e}")
                height, width = target_image.shape[:2]
                start_time = time.time()
                results.append(SegmentationResult(
                    masks=[],
                    count=0,
                    processing_time_ms=0.0,
                    image_size=(width, height),
                ))

        return results



    def _process_semantic_results(
        self,
        results,
        image: np.ndarray,
        start_time: float,
        labels: Optional[List[str]] = None,
    ) -> SegmentationResult:
        """处理 SAM3SemanticPredictor 的结果
        
        Args:
            results: 预测器返回的结果
            image: 原始图像
            start_time: 开始时间戳
            labels: 可选的标签列表（用于文本提示）
            
        Returns:
            SegmentationResult: 处理后的分割结果
        """
        masks_data: List[MaskData] = []
        
        # 获取原图尺寸
        orig_height, orig_width = image.shape[:2]
        
        for r in results:
            if r.masks is not None:
                for i, mask in enumerate(r.masks.data):
                    # 转换掩码为 numpy 数组
                    mask_np = mask.cpu().numpy().astype(np.uint8) * 255
                    
                    # 缩放 mask 到原图尺寸
                    mask_h, mask_w = mask_np.shape[:2]
                    if mask_h != orig_height or mask_w != orig_width:
                        mask_np = cv2.resize(
                            mask_np,
                            (orig_width, orig_height),
                            interpolation=cv2.INTER_NEAREST
                        )
                    
                    # 编码为 Base64 PNG
                    mask_base64 = self._encode_mask_to_base64(mask_np)
                    
                    # 获取边界框
                    if r.boxes is not None and i < len(r.boxes.xyxy):
                        bbox = r.boxes.xyxy[i].cpu().numpy().tolist()
                        score = float(r.boxes.conf[i].cpu().numpy())
                    else:
                        # 从掩码计算边界框
                        bbox = self._compute_bbox_from_mask(mask_np)
                        score = 1.0
                    
                    # 计算掩码面积
                    area = int(np.sum(mask_np > 0))
                    
                    # 确定标签
                    label = None
                    if labels and i < len(labels):
                        label = labels[i]
                    
                    masks_data.append(MaskData(
                        mask_base64=mask_base64,
                        bbox=bbox,
                        score=score,
                        label=label,
                        area=area,
                    ))
        
        # 计算处理时间
        processing_time_ms = (time.time() - start_time) * 1000
        
        logger.info(
            f"Segmentation completed: {len(masks_data)} masks, "
            f"{processing_time_ms:.2f}ms"
        )
        
        return SegmentationResult(
            masks=masks_data,
            count=len(masks_data),
            processing_time_ms=processing_time_ms,
            image_size=(orig_width, orig_height),
        )

    def _process_point_results(
        self,
        results,
        image: np.ndarray,
        start_time: float,
    ) -> SegmentationResult:
        """处理 SAM 点击模型的结果
        
        Args:
            results: 模型返回的结果
            image: 原始图像
            start_time: 开始时间戳
            
        Returns:
            SegmentationResult: 处理后的分割结果
        """
        masks_data: List[MaskData] = []
        
        # 获取原图尺寸
        orig_height, orig_width = image.shape[:2]
        
        for r in results:
            if r.masks is not None:
                for i, mask in enumerate(r.masks.data):
                    # 转换掩码为 numpy 数组
                    mask_np = mask.cpu().numpy().astype(np.uint8) * 255
                    
                    # 缩放 mask 到原图尺寸
                    mask_h, mask_w = mask_np.shape[:2]
                    if mask_h != orig_height or mask_w != orig_width:
                        mask_np = cv2.resize(
                            mask_np,
                            (orig_width, orig_height),
                            interpolation=cv2.INTER_NEAREST
                        )
                    
                    # 编码为 Base64 PNG
                    mask_base64 = self._encode_mask_to_base64(mask_np)
                    
                    # 获取边界框
                    if r.boxes is not None and i < len(r.boxes.xyxy):
                        bbox = r.boxes.xyxy[i].cpu().numpy().tolist()
                        score = float(r.boxes.conf[i].cpu().numpy())
                    else:
                        # 从掩码计算边界框
                        bbox = self._compute_bbox_from_mask(mask_np)
                        score = 1.0
                    
                    # 计算掩码面积
                    area = int(np.sum(mask_np > 0))
                    
                    masks_data.append(MaskData(
                        mask_base64=mask_base64,
                        bbox=bbox,
                        score=score,
                        label=None,  # 点击分割没有标签
                        area=area,
                    ))
        
        # 计算处理时间
        processing_time_ms = (time.time() - start_time) * 1000
        
        logger.info(
            f"Point segmentation completed: {len(masks_data)} masks, "
            f"{processing_time_ms:.2f}ms"
        )
        
        return SegmentationResult(
            masks=masks_data,
            count=len(masks_data),
            processing_time_ms=processing_time_ms,
            image_size=(orig_width, orig_height),
        )
    
    def _encode_mask_to_base64(self, mask: np.ndarray) -> str:
        """将掩码编码为 Base64 PNG 字符串
        
        Args:
            mask: 二值掩码数组 (H, W)，值为 0 或 255
            
        Returns:
            Base64 编码的 PNG 字符串
        """
        # 使用 OpenCV 编码为 PNG
        success, encoded = cv2.imencode('.png', mask)
        if not success:
            raise ValueError("Failed to encode mask to PNG")
        
        # 转换为 Base64
        return base64.b64encode(encoded).decode('utf-8')
    
    def _compute_bbox_from_mask(self, mask: np.ndarray) -> List[float]:
        """从掩码计算边界框
        
        Args:
            mask: 二值掩码数组 (H, W)
            
        Returns:
            边界框坐标 [x1, y1, x2, y2]
        """
        # 找到非零像素的位置
        rows = np.any(mask > 0, axis=1)
        cols = np.any(mask > 0, axis=0)
        
        if not np.any(rows) or not np.any(cols):
            return [0.0, 0.0, 0.0, 0.0]
        
        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]
        
        return [float(x_min), float(y_min), float(x_max), float(y_max)]


# 全局分割服务实例
_segmentation_service: Optional[SegmentationService] = None


def get_segmentation_service() -> SegmentationService:
    """获取全局分割服务实例
    
    Returns:
        SegmentationService 实例
    """
    global _segmentation_service
    if _segmentation_service is None:
        _segmentation_service = SegmentationService()
    return _segmentation_service


def parse_text_prompts(text: str) -> List[str]:
    """解析逗号分隔的文本提示
    
    将逗号分隔的文本字符串解析为独立的提示词列表。
    
    Args:
        text: 逗号分隔的文本字符串，如 "person, car, dog"
        
    Returns:
        提示词列表，如 ["person", "car", "dog"]
        
    Example:
        >>> parse_text_prompts("person, car, dog")
        ["person", "car", "dog"]
        >>> parse_text_prompts("  person  ,  car  ")
        ["person", "car"]
        >>> parse_text_prompts("")
        []
    """
    if not text:
        return []
    
    # 分割、去除空白、过滤空字符串
    prompts = [p.strip() for p in text.split(",")]
    prompts = [p for p in prompts if p]
    
    return prompts
