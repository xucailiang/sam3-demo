import axios, { AxiosError } from 'axios';
import type {
  PointPrompt,
  BoxPrompt,
  SegmentationResult,
  BatchSegmentationResult,
  MaskData,
} from '../types';

const api = axios.create({
  baseURL: '/api',
  timeout: 30000,
});

/**
 * Extract a human-readable error message from an API or network error.
 */
function extractErrorMessage(err: unknown): string {
  if (err instanceof AxiosError) {
    // Network-level failure (no response received)
    if (!err.response) {
      if (err.code === 'ECONNABORTED') {
        return '请求超时，请稍后重试';
      }
      return '网络连接失败，请检查网络或后端服务是否运行';
    }
    // Server returned an error response — try to use the envelope message
    const data = err.response.data as Record<string, unknown> | undefined;
    if (data && typeof data.error === 'string') {
      return data.error;
    }
    // Fallback to status text
    return `服务器错误 (${err.response.status})`;
  }
  if (err instanceof Error) {
    return err.message;
  }
  return '未知错误';
}

/**
 * Map a backend mask object (snake_case) to the frontend MaskData type (camelCase).
 */
function mapMask(raw: Record<string, unknown>): MaskData {
  return {
    maskBase64: raw.mask_base64 as string,
    bbox: raw.bbox as [number, number, number, number],
    score: raw.score as number,
    label: (raw.label as string | undefined) ?? undefined,
    area: raw.area as number,
    color: '', // assigned later by the frontend rendering layer
  };
}

/**
 * Map a backend segmentation result (snake_case) to the frontend type (camelCase).
 */
function mapResult(raw: Record<string, unknown>): SegmentationResult {
  const masks = raw.masks as Record<string, unknown>[];
  return {
    masks: masks.map(mapMask),
    count: raw.count as number,
    processingTimeMs: raw.processing_time_ms as number,
    imageSize: raw.image_size as [number, number],
  };
}

/**
 * Parse the API response envelope and return the mapped result.
 */
function parseResponse(data: Record<string, unknown>): SegmentationResult {
  if (!data.success) {
    throw new Error((data.error as string) || 'Segmentation failed');
  }
  if (!data.data) {
    throw new Error('No data in response');
  }
  return mapResult(data.data as Record<string, unknown>);
}

/**
 * Text prompt segmentation.
 *
 * Sends the image and comma-separated text prompts to the backend.
 * Requirements: 2.1
 */
export async function segmentWithText(
  image: File,
  prompts: string,
): Promise<SegmentationResult> {
  const formData = new FormData();
  formData.append('file', image);
  formData.append('prompts', prompts);

  try {
    const { data } = await api.post('/segment/text', formData);
    return parseResponse(data);
  } catch (err) {
    throw new Error(extractErrorMessage(err));
  }
}

/**
 * Point prompt segmentation.
 *
 * Sends the image along with point coordinates and labels.
 * Requirements: 3.4
 */
export async function segmentWithPoints(
  image: File,
  points: PointPrompt[],
): Promise<SegmentationResult> {
  const formData = new FormData();
  formData.append('file', image);
  formData.append('points', JSON.stringify(points.map((p) => [p.x, p.y])));
  formData.append('labels', JSON.stringify(points.map((p) => p.label)));

  try {
    const { data } = await api.post('/segment/points', formData);
    return parseResponse(data);
  } catch (err) {
    throw new Error(extractErrorMessage(err));
  }
}

/**
 * Bounding box prompt segmentation.
 *
 * Sends the image along with bounding box coordinates.
 * Requirements: 4.3
 */
export async function segmentWithBoxes(
  image: File,
  boxes: BoxPrompt[],
): Promise<SegmentationResult> {
  const formData = new FormData();
  formData.append('file', image);
  formData.append(
    'boxes',
    JSON.stringify(boxes.map((b) => [b.x1, b.y1, b.x2, b.y2])),
  );

  try {
    const { data } = await api.post('/segment/boxes', formData);
    return parseResponse(data);
  } catch (err) {
    throw new Error(extractErrorMessage(err));
  }
}


/**
 * Stitch segmentation (single image).
 *
 * Sends the target image, a sample image, and a bounding box defining
 * the sample region to the backend. The backend stitches the images
 * horizontally and runs SAM3 inference.
 * Requirements: 8.1
 */
export async function segmentWithStitch(
  targetImage: File,
  sampleImage: File,
  sampleBox: BoxPrompt,
): Promise<SegmentationResult> {
  const formData = new FormData();
  formData.append('file', targetImage);
  formData.append('sample_file', sampleImage);
  formData.append(
    'sample_box',
    JSON.stringify([sampleBox.x1, sampleBox.y1, sampleBox.x2, sampleBox.y2]),
  );

  try {
    const { data } = await api.post('/segment/stitch', formData);
    return parseResponse(data);
  } catch (err) {
    throw new Error(extractErrorMessage(err));
  }
}


/**
 * Parse a batch API response envelope and return the mapped result.
 */
function parseBatchResponse(data: Record<string, unknown>): BatchSegmentationResult {
  if (!data.success) {
    throw new Error((data.error as string) || 'Batch segmentation failed');
  }
  if (!data.data) {
    throw new Error('No data in batch response');
  }
  const raw = data.data as Record<string, unknown>;
  const rawResults = raw.results as Record<string, unknown>[];
  return {
    results: rawResults.map(mapResult),
    total: raw.total as number,
    successCount: raw.success_count as number,
    failedCount: raw.failed_count as number,
    totalProcessingTimeMs: raw.total_processing_time_ms as number,
  };
}

/**
 * Batch text prompt segmentation.
 *
 * Sends multiple images with the same text prompts.
 * Requirements: 8.2
 */
export async function batchSegmentWithText(
  files: File[],
  prompts: string,
): Promise<BatchSegmentationResult> {
  const formData = new FormData();
  for (const file of files) {
    formData.append('files', file);
  }
  formData.append('prompts', prompts);

  try {
    const { data } = await api.post('/batch/text', formData);
    return parseBatchResponse(data);
  } catch (err) {
    throw new Error(extractErrorMessage(err));
  }
}

/**
 * Batch stitch segmentation.
 *
 * Sends multiple target images with a sample image and region.
 * The backend stitches each target with the sample and runs SAM3 inference.
 * Requirements: 8.2
 */
export async function batchSegmentWithStitch(
  files: File[],
  sampleImage: File,
  sampleBox: BoxPrompt,
): Promise<BatchSegmentationResult> {
  const formData = new FormData();
  for (const file of files) {
    formData.append('files', file);
  }
  formData.append('sample_file', sampleImage);
  formData.append(
    'sample_box',
    JSON.stringify([sampleBox.x1, sampleBox.y1, sampleBox.x2, sampleBox.y2]),
  );

  try {
    const { data } = await api.post('/batch/stitch', formData);
    return parseBatchResponse(data);
  } catch (err) {
    throw new Error(extractErrorMessage(err));
  }
}
