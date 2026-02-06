import { useCallback, useState } from 'react';
import type { PointPrompt, BoxPrompt, SegmentationResult } from '../types';
import {
  segmentWithText as apiSegmentWithText,
  segmentWithPoints as apiSegmentWithPoints,
  segmentWithBoxes as apiSegmentWithBoxes,
  segmentWithStitch as apiSegmentWithStitch,
} from '../api/client';

export interface UseSegmentationReturn {
  result: SegmentationResult | null;
  isLoading: boolean;
  error: string | null;
  segmentWithText: (image: File, prompts: string) => Promise<void>;
  segmentWithPoints: (image: File, points: PointPrompt[]) => Promise<void>;
  segmentWithBoxes: (image: File, boxes: BoxPrompt[]) => Promise<void>;
  segmentWithStitch: (targetImage: File, sampleImage: File, sampleBox: BoxPrompt) => Promise<void>;
  clearResult: () => void;
}

/**
 * Hook that manages segmentation request state, loading, and errors.
 *
 * Wraps the API client functions and exposes a consistent interface
 * for text, point, and box segmentation modes.
 *
 * Requirements: 2.3, 3.4, 4.3
 */
export function useSegmentation(): UseSegmentationReturn {
  const [result, setResult] = useState<SegmentationResult | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const run = useCallback(
    async (fn: () => Promise<SegmentationResult>) => {
      setIsLoading(true);
      setError(null);
      try {
        const res = await fn();
        setResult(res);
      } catch (err: unknown) {
        const message =
          err instanceof Error ? err.message : 'Unknown error occurred';
        setError(message);
        setResult(null);
      } finally {
        setIsLoading(false);
      }
    },
    [],
  );

  const segmentWithText = useCallback(
    (image: File, prompts: string) =>
      run(() => apiSegmentWithText(image, prompts)),
    [run],
  );

  const segmentWithPoints = useCallback(
    (image: File, points: PointPrompt[]) =>
      run(() => apiSegmentWithPoints(image, points)),
    [run],
  );

  const segmentWithBoxes = useCallback(
    (image: File, boxes: BoxPrompt[]) =>
      run(() => apiSegmentWithBoxes(image, boxes)),
    [run],
  );

  const segmentWithStitch = useCallback(
    (targetImage: File, sampleImage: File, sampleBox: BoxPrompt) =>
      run(() => apiSegmentWithStitch(targetImage, sampleImage, sampleBox)),
    [run],
  );

  const clearResult = useCallback(() => {
    setResult(null);
    setError(null);
  }, []);

  return {
    result,
    isLoading,
    error,
    segmentWithText,
    segmentWithPoints,
    segmentWithBoxes,
    segmentWithStitch,
    clearResult,
  };
}
