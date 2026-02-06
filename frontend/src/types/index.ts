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
