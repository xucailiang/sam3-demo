import { useCallback, useRef, useState } from 'react';
import type { BoxPrompt, BatchProgress, BatchResult } from '../types';
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
    async (files: File[], sampleImage: File, sampleBox: BoxPrompt) => {
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
    cancelBatch,
    clearResults,
  };
}
