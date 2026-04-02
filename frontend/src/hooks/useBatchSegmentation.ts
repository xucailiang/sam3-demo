import { useCallback, useRef, useState } from 'react';
import type { BoxPrompt, BatchProgress, BatchResult, SegmentationResult, SampleInferResponse } from '../types';
import {
  batchSegmentWithText as apiBatchText,
  batchSegmentWithStitch as apiBatchStitch,
} from '../api/client';

export interface UseBatchSegmentationReturn {
  results: BatchResult[];
  progress: BatchProgress;
  isProcessing: boolean;
  batchSegmentWithText: (files: File[], prompts: string) => Promise<void>;
  batchSegmentWithStitch: (
    files: File[],
    sampleImage: File,
    sampleBox: BoxPrompt,
    confidence?: number,
  ) => Promise<void>;
  batchSegmentWithSample: (
    files: File[],
    inferFn: (files: File[]) => Promise<SampleInferResponse | null>,
  ) => Promise<void>;
  cancelBatch: () => void;
  clearResults: () => void;
}

const IDLE_PROGRESS: BatchProgress = {
  total: 0,
  completed: 0,
  currentIndex: 0,
  status: 'idle',
};

/**
 * Hook that manages batch segmentation state and progress.
 *
 * Requirements: 8.2, 8.3, 8.4
 */
export function useBatchSegmentation(): UseBatchSegmentationReturn {
  const [results, setResults] = useState<BatchResult[]>([]);
  const [progress, setProgress] = useState<BatchProgress>(IDLE_PROGRESS);
  const [isProcessing, setIsProcessing] = useState(false);
  const cancelledRef = useRef(false);

  const batchSegmentWithText = useCallback(
    async (files: File[], prompts: string) => {
      cancelledRef.current = false;
      setIsProcessing(true);
      setResults([]);
      setProgress({
        total: files.length,
        completed: 0,
        currentIndex: 0,
        status: 'processing',
      });

      try {
        const batchResult = await apiBatchText(files, prompts);

        if (cancelledRef.current) return;

        const mapped: BatchResult[] = files.map((file, i) => ({
          file,
          result: batchResult.results[i] ?? null,
          error: null,
        }));

        setResults(mapped);
        setProgress({
          total: files.length,
          completed: files.length,
          currentIndex: files.length,
          status: 'completed',
        });
      } catch (err: unknown) {
        if (cancelledRef.current) return;
        const message = err instanceof Error ? err.message : 'Batch failed';
        setResults(
          files.map((file) => ({ file, result: null, error: message })),
        );
        setProgress((prev) => ({ ...prev, status: 'error' }));
      } finally {
        setIsProcessing(false);
      }
    },
    [],
  );

  const batchSegmentWithStitch = useCallback(
    async (files: File[], sampleImage: File, sampleBox: BoxPrompt, confidence: number = 0.25) => {
      cancelledRef.current = false;
      setIsProcessing(true);
      setResults([]);
      setProgress({
        total: files.length,
        completed: 0,
        currentIndex: 0,
        status: 'processing',
      });

      try {
        const batchResult = await apiBatchStitch(
          files,
          sampleImage,
          sampleBox,
          confidence,
        );

        if (cancelledRef.current) return;

        const mapped: BatchResult[] = files.map((file, i) => ({
          file,
          result: batchResult.results[i] ?? null,
          error: null,
        }));

        setResults(mapped);
        setProgress({
          total: files.length,
          completed: files.length,
          currentIndex: files.length,
          status: 'completed',
        });
      } catch (err: unknown) {
        if (cancelledRef.current) return;
        const message = err instanceof Error ? err.message : 'Batch failed';
        setResults(
          files.map((file) => ({ file, result: null, error: message })),
        );
        setProgress((prev) => ({ ...prev, status: 'error' }));
      } finally {
        setIsProcessing(false);
      }
    },
    [],
  );

  /**
   * Batch segmentation using sample-based feature caching.
   * Requirements: 1.2, 1.4, 5.1, 5.2
   */
  const batchSegmentWithSample = useCallback(
    async (
      files: File[],
      inferFn: (files: File[]) => Promise<SampleInferResponse | null>,
    ) => {
      cancelledRef.current = false;
      setIsProcessing(true);
      setResults([]);
      setProgress({
        total: files.length,
        completed: 0,
        currentIndex: 0,
        status: 'processing',
      });

      try {
        const response = await inferFn(files);

        if (cancelledRef.current) return;

        if (!response) {
          throw new Error('Sample inference failed');
        }

        // Convert SampleInferResponse to BatchResult format
        const mapped: BatchResult[] = files.map((file, i) => {
          const inferResult = response.results[i];
          if (!inferResult || inferResult.error) {
            return {
              file,
              result: null,
              error: inferResult?.error || 'Inference failed',
            };
          }
          // Convert to SegmentationResult format
          const segResult: SegmentationResult = {
            masks: inferResult.masks,
            count: inferResult.count,
            processingTimeMs: inferResult.processing_time_ms,
            imageSize: inferResult.image_size,
          };
          return {
            file,
            result: segResult,
            error: null,
          };
        });

        setResults(mapped);
        setProgress({
          total: files.length,
          completed: files.length,
          currentIndex: files.length,
          status: 'completed',
        });
      } catch (err: unknown) {
        if (cancelledRef.current) return;
        const message = err instanceof Error ? err.message : 'Batch failed';
        setResults(
          files.map((file) => ({ file, result: null, error: message })),
        );
        setProgress((prev) => ({ ...prev, status: 'error' }));
      } finally {
        setIsProcessing(false);
      }
    },
    [],
  );

  const cancelBatch = useCallback(() => {
    cancelledRef.current = true;
    setIsProcessing(false);
    setProgress((prev) => ({ ...prev, status: 'idle' }));
  }, []);

  const clearResults = useCallback(() => {
    setResults([]);
    setProgress(IDLE_PROGRESS);
  }, []);

  return {
    results,
    progress,
    isProcessing,
    batchSegmentWithText,
    batchSegmentWithStitch,
    batchSegmentWithSample,
    cancelBatch,
    clearResults,
  };
}
