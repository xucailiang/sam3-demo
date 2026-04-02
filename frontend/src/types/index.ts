// SAM3 Demo Type Definitions

/** Interaction mode for the canvas */
export type InteractionMode = 'text' | 'point' | 'box';

/** Workflow mode for SampleWorkflow component */
export type WorkflowMode = 'stitch' | 'text';

/** Point prompt with foreground/background label */
export interface PointPrompt {
  x: number;
  y: number;
  label: 0 | 1; // 0=background, 1=foreground
}

/** Bounding box prompt [top-left to bottom-right] */
export interface BoxPrompt {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

/** Image example prompt: an example image with a region of interest */
export interface ImageExamplePrompt {
  exampleFile: File;
  exampleBox: BoxPrompt;
}

/** Single mask data returned from the backend */
export interface MaskData {
  maskBase64: string;
  bbox: [number, number, number, number];
  score: number;
  label?: string;
  area: number;
  color: string; // display color assigned by frontend
  /** 缺陷类别标签（样本推理时返回） */
  category?: string;
}

/** Segmentation result for a single image */
export interface SegmentationResult {
  masks: MaskData[];
  count: number;
  processingTimeMs: number;
  imageSize: [number, number];
}

/** Standard API response wrapper */
export interface SegmentationResponse {
  success: boolean;
  data?: SegmentationResult;
  error?: string;
}

/** Batch processing progress */
export interface BatchProgress {
  total: number;
  completed: number;
  currentIndex: number;
  status: 'idle' | 'processing' | 'completed' | 'error';
}

/** Result for a single image in a batch */
export interface BatchResult {
  file: File;
  result: SegmentationResult | null;
  error: string | null;
}

/** Batch segmentation result from the backend */
export interface BatchSegmentationResult {
  results: SegmentationResult[];
  total: number;
  successCount: number;
  failedCount: number;
  totalProcessingTimeMs: number;
}

/** Batch API response wrapper */
export interface BatchSegmentationResponse {
  success: boolean;
  data?: BatchSegmentationResult;
  error?: string;
}

// ============================================
// Mode A Result Filtering Types
// ============================================

/** 带选择状态的掩码数据 */
export interface SelectableMaskData extends MaskData {
  /** 原始索引（用于追踪） */
  originalIndex: number;
  /** 是否被选中 */
  isSelected: boolean;
  /** 是否通过过滤（高于阈值） */
  passesFilter: boolean;
}

/** 过滤结果统计 */
export interface FilterStats {
  /** 总检测数量 */
  totalCount: number;
  /** 过滤后数量（高于阈值） */
  filteredCount: number;
  /** 选中数量 */
  selectedCount: number;
}

/** 单张图片的选择状态 */
export interface ImageSelectionState {
  /** 掩码选择状态 Map */
  selectionMap: Map<number, boolean>;
  /** 最后更新时间 */
  lastUpdated: number;
}

/** 批量结果的选择状态 */
export interface BatchSelectionState {
  /** 图片索引 -> 选择状态 */
  imageStates: Map<number, ImageSelectionState>;
  /** 全局置信度阈值 */
  globalThreshold: number;
}

// ============================================
// Multi-BBox Defect Annotation Types
// ============================================

/** 缺陷边界框（带类别标签） */
export interface DefectBox {
  /** 唯一标识 */
  id: string;
  /** 左上角 X 坐标 */
  x1: number;
  /** 左上角 Y 坐标 */
  y1: number;
  /** 右下角 X 坐标 */
  x2: number;
  /** 右下角 Y 坐标 */
  y2: number;
  /** 缺陷类别标签 */
  category: string;
  /** 显示颜色 */
  color: string;
}

/** 创建样本响应 */
export interface CreateSampleResponse {
  /** 样本唯一标识 */
  sample_id: string;
  /** BBox 数量 */
  boxes_count: number;
  /** 特征提取耗时（毫秒） */
  feature_time_ms: number;
}

/** 单张图片的推理结果 */
export interface SampleInferResult {
  /** 分割掩码列表 */
  masks: MaskData[];
  /** 检测数量 */
  count: number;
  /** 处理时间（毫秒） */
  processing_time_ms: number;
  /** 图像尺寸 [宽, 高] */
  image_size: [number, number];
  /** 错误信息（如果有） */
  error?: string;
}

/** 样本推理响应 */
export interface SampleInferResponse {
  /** 推理结果列表 */
  results: SampleInferResult[];
  /** 总图片数 */
  total: number;
  /** 成功数量 */
  success_count: number;
  /** 失败数量 */
  failed_count: number;
  /** 总处理时间（毫秒） */
  total_time_ms: number;
}

