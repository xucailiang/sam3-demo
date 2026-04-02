/**
 * useSampleApi - Hook for sample-based feature caching API
 *
 * Provides functions to create samples with cached features,
 * perform batch inference using cached features, and delete samples.
 *
 * Requirements: 4.1, 4.2, 4.5
 */
import { useCallback, useState } from 'react';
import axios, { AxiosError } from 'axios';
import type {
  DefectBox,
  CreateSampleResponse,
  SampleInferResponse,
  MaskData,
} from '../types';

const api = axios.create({
  baseURL: '/api',
  timeout: 120000, // 2 minutes for large batch operations
});

/**
 * Extract a human-readable error message from an API or network error.
 */
function extractErrorMessage(err: unknown): string {
  if (err instanceof AxiosError) {
    if (!err.response) {
      if (err.code === 'ECONNABORTED') {
        return '请求超时，请稍后重试';
      }
      return '网络连接失败，请检查网络或后端服务是否运行';
    }
    const data = err.response.data as Record<string, unknown> | undefined;
    if (data && typeof data.error === 'string') {
      return data.error;
    }
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
    category: (raw.category as string | undefined) ?? undefined,
  };
}

export interface UseSampleApiReturn {
  /** Current sample ID (null if no sample created) */
  sampleId: string | null;
  /** Feature extraction time in ms */
  featureTimeMs: number | null;
  /** Loading state for create/delete operations */
  isLoading: boolean;
  /** Error message if any */
  error: string | null;
  /** Create a sample with cached features */
  createSample: (image: File, boxes: DefectBox[]) => Promise<string | null>;
  /** Perform batch inference using cached features */
  inferWithSample: (
    sampleId: string,
    files: File[],
    confidence?: number
  ) => Promise<SampleInferResponse | null>;
  /** Delete a sample and release cached features */
  deleteSample: (sampleId: string) => Promise<boolean>;
  /** Clear current sample state */
  clearSample: () => void;
}

/**
 * Hook for sample-based feature caching API operations.
 *
 * Provides functions to:
 * - Create samples with cached features (Requirements 4.1, 4.3)
 * - Perform batch inference using cached features (Requirements 4.2, 4.4)
 * - Delete samples and release cache (Requirements 4.5)
 */
export function useSampleApi(): UseSampleApiReturn {
  const [sampleId, setSampleId] = useState<string | null>(null);
  const [featureTimeMs, setFeatureTimeMs] = useState<number | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  /**
   * Create a sample with cached features.
   * Requirements: 4.1, 4.3
   */
  const createSample = useCallback(
    async (image: File, boxes: DefectBox[]): Promise<string | null> => {
      setIsLoading(true);
      setError(null);

      try {
        const formData = new FormData();
        formData.append('image', image);
        // Convert DefectBox to backend format (without id and color)
        const boxesData = boxes.map((b) => ({
          x1: b.x1,
          y1: b.y1,
          x2: b.x2,
          y2: b.y2,
          category: b.category,
        }));
        formData.append('boxes', JSON.stringify(boxesData));

        const { data } = await api.post('/sample/create', formData);

        if (!data.success) {
          throw new Error(data.error || 'Failed to create sample');
        }

        const result = data.data as CreateSampleResponse;
        setSampleId(result.sample_id);
        setFeatureTimeMs(result.feature_time_ms);
        return result.sample_id;
      } catch (err) {
        const message = extractErrorMessage(err);
        setError(message);
        return null;
      } finally {
        setIsLoading(false);
      }
    },
    []
  );

  /**
   * Perform batch inference using cached features.
   * Requirements: 4.2, 4.4, 5.1, 5.2
   */
  const inferWithSample = useCallback(
    async (
      targetSampleId: string,
      files: File[],
      confidence: number = 0.25
    ): Promise<SampleInferResponse | null> => {
      setError(null);

      try {
        const formData = new FormData();
        formData.append('sample_id', targetSampleId);
        for (const file of files) {
          formData.append('files', file);
        }
        formData.append('confidence', confidence.toString());

        const { data } = await api.post('/sample/infer', formData);

        if (!data.success) {
          throw new Error(data.error || 'Failed to infer with sample');
        }

        // Map backend response to frontend types
        const rawData = data.data as Record<string, unknown>;
        const rawResults = rawData.results as Record<string, unknown>[];

        const response: SampleInferResponse = {
          results: rawResults.map((r) => {
            const masks = r.masks as Record<string, unknown>[];
            return {
              masks: masks.map(mapMask),
              count: r.count as number,
              processing_time_ms: r.processing_time_ms as number,
              image_size: r.image_size as [number, number],
              error: (r.error as string | undefined) ?? undefined,
            };
          }),
          total: rawData.total as number,
          success_count: rawData.success_count as number,
          failed_count: rawData.failed_count as number,
          total_time_ms: rawData.total_time_ms as number,
        };

        return response;
      } catch (err) {
        const message = extractErrorMessage(err);
        setError(message);
        return null;
      }
    },
    []
  );

  /**
   * Delete a sample and release cached features.
   * Requirements: 4.5
   */
  const deleteSample = useCallback(
    async (targetSampleId: string): Promise<boolean> => {
      setIsLoading(true);
      setError(null);

      try {
        const { data } = await api.delete(`/sample/${targetSampleId}`);

        if (!data.success) {
          throw new Error(data.error || 'Failed to delete sample');
        }

        // Clear local state if deleting current sample
        if (targetSampleId === sampleId) {
          setSampleId(null);
          setFeatureTimeMs(null);
        }

        return true;
      } catch (err) {
        const message = extractErrorMessage(err);
        setError(message);
        return false;
      } finally {
        setIsLoading(false);
      }
    },
    [sampleId]
  );

  /**
   * Clear current sample state.
   */
  const clearSample = useCallback(() => {
    setSampleId(null);
    setFeatureTimeMs(null);
    setError(null);
  }, []);

  return {
    sampleId,
    featureTimeMs,
    isLoading,
    error,
    createSample,
    inferWithSample,
    deleteSample,
    clearSample,
  };
}
