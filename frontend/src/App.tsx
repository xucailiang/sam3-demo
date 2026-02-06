import { useState, useCallback, useRef } from 'react';
import './App.css';
import type { InteractionMode, PointPrompt, BoxPrompt, MaskData, SegmentationResult } from './types';
import { ImageCanvas } from './components/ImageCanvas';
import { PromptPanel } from './components/PromptPanel';
import { ResultPanel } from './components/ResultPanel';
import { BatchPanel } from './components/BatchPanel';
import { SampleWorkflow } from './components/SampleWorkflow';
import { ErrorBanner } from './components/ErrorBanner';
import { useSegmentation } from './hooks/useSegmentation';
import { validateFileExtension } from './utils/validation';
import { assignMaskColors } from './utils/canvas';
import { clampOpacity, toggleMaskVisibility } from './utils/displayState';
import { exportMaskAsPNG, exportOverlayAsPNG, exportResultAsJSON } from './utils/export';

/**
 * Main application component.
 *
 * Assembles ImageCanvas, PromptPanel, and ResultPanel.
 * Manages application state: image, interaction mode, prompts, results, display settings.
 *
 * Requirements: 1.1, 1.2, 1.3, 1.4, 2.3, 5.2, 7.1, 7.3, 7.4
 */
function App() {
  // Image state
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [imageEl, setImageEl] = useState<HTMLImageElement | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // App mode: single image vs batch vs sample workflow
  const [appMode, setAppMode] = useState<'sample-workflow' | 'single' | 'batch'>('sample-workflow');

  // Interaction state
  const [mode, setMode] = useState<InteractionMode>('text');
  const [textPrompt, setTextPrompt] = useState('');
  const [points, setPoints] = useState<PointPrompt[]>([]);
  const [boxes, setBoxes] = useState<BoxPrompt[]>([]);

  // Display state
  const [maskOpacity, setMaskOpacity] = useState(0.5);
  const [showMasks, setShowMasks] = useState(true);
  const [hoveredMaskIndex, setHoveredMaskIndex] = useState<number | null>(null);

  // Upload error
  const [uploadError, setUploadError] = useState<string | null>(null);

  // Segmentation hook
  const { result, isLoading, error, segmentWithText, segmentWithPoints, segmentWithBoxes, clearResult } =
    useSegmentation();

  // Derive colored masks from result
  const coloredMasks: MaskData[] = result
    ? (() => {
        const colors = assignMaskColors(result.masks.length);
        return result.masks.map((m, i) => ({ ...m, color: colors[i] }));
      })()
    : [];

  // --- Image upload ---
  const handleFileChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    if (!validateFileExtension(file.name)) {
      setUploadError('不支持的文件格式，请上传 jpg、jpeg、png 或 webp 图像');
      return;
    }

    setUploadError(null);
    setImageFile(file);
    clearResult();
    setPoints([]);
    setBoxes([]);

    const url = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () => {
      setImageEl(img);
    };
    img.src = url;
  }, [clearResult]);

  // --- Prompt handlers ---
  const handlePointAdd = useCallback((point: PointPrompt) => {
    setPoints((prev) => [...prev, point]);
  }, []);

  const handleBoxComplete = useCallback((box: BoxPrompt) => {
    setBoxes((prev) => [...prev, box]);
  }, []);

  const handleClearPoints = useCallback(() => {
    setPoints([]);
  }, []);

  const handleClearBoxes = useCallback(() => {
    setBoxes([]);
  }, []);

  // --- Submit segmentation ---
  const handleSubmit = useCallback(async () => {
    if (!imageFile) return;

    if (mode === 'text') {
      if (!textPrompt.trim()) return;
      await segmentWithText(imageFile, textPrompt);
    } else if (mode === 'point') {
      if (points.length === 0) return;
      await segmentWithPoints(imageFile, points);
    } else if (mode === 'box') {
      if (boxes.length === 0) return;
      await segmentWithBoxes(imageFile, boxes);
    }
  }, [imageFile, mode, textPrompt, points, boxes, segmentWithText, segmentWithPoints, segmentWithBoxes]);

  // --- Display handlers ---
  const handleOpacityChange = useCallback((value: number) => {
    setMaskOpacity(clampOpacity(value));
  }, []);

  const handleToggleMasks = useCallback(() => {
    setShowMasks((prev) => toggleMaskVisibility(prev));
  }, []);

  const handleMaskHover = useCallback((index: number | null) => {
    setHoveredMaskIndex(index);
  }, []);

  // Export handlers (Requirements: 9.1, 9.2, 9.3)
  const handleExportMask = useCallback(() => {
    if (!result || coloredMasks.length === 0) return;
    // Export the first mask; for multi-mask export each could be triggered individually
    exportMaskAsPNG(coloredMasks[0], result.imageSize);
  }, [result, coloredMasks]);

  const handleExportOverlay = useCallback(() => {
    if (!result || !imageEl || coloredMasks.length === 0) return;
    exportOverlayAsPNG(imageEl, coloredMasks, maskOpacity);
  }, [result, imageEl, coloredMasks, maskOpacity]);

  const handleExportJSON = useCallback(() => {
    if (!result) return;
    exportResultAsJSON({ ...result, masks: coloredMasks });
  }, [result, coloredMasks]);

  // Determine if segmentation controls should be enabled
  const canSegment = imageEl !== null;

  // Handle viewing a batch result detail
  const handleViewBatchResult = useCallback((_batchResult: SegmentationResult, file: File) => {
    // Switch to single mode and display the result
    setAppMode('single');
    setImageFile(file);
    clearResult();

    const url = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () => {
      setImageEl(img);
      // We can't directly set the segmentation result via the hook,
      // so we just load the image for viewing
    };
    img.src = url;
  }, [clearResult]);

  return (
    <div className="app">
      <header className="app-header">
        <h1>SAM3 Demo</h1>
        <p>图像分割演示项目</p>
        {/* Mode tabs: single vs batch */}
        <div style={{ display: 'flex', gap: '0.5rem', justifyContent: 'center', marginTop: '0.5rem' }}>
          {([
            { key: 'sample-workflow' as const, label: '样品标注工作流' },
            { key: 'single' as const, label: '单图模式' },
            { key: 'batch' as const, label: '批量模式' },
          ]).map((m) => (
            <button
              key={m.key}
              onClick={() => setAppMode(m.key)}
              style={{
                padding: '0.4rem 1rem',
                borderRadius: '4px',
                border: appMode === m.key ? '2px solid #4ECDC4' : '1px solid #555',
                background: appMode === m.key ? '#4ECDC4' : 'transparent',
                color: appMode === m.key ? '#000' : 'inherit',
                cursor: 'pointer',
                fontWeight: appMode === m.key ? 600 : 400,
              }}
            >
              {m.label}
            </button>
          ))}
        </div>
      </header>

      <div className="app-body">
        {appMode === 'sample-workflow' ? (
          /* Sample workflow mode */
          <div style={{ width: '100%', maxWidth: 800, margin: '0 auto' }}>
            <SampleWorkflow />
          </div>
        ) : appMode === 'batch' ? (
          /* Batch mode */
          <div className="app-sidebar" style={{ width: '100%', maxWidth: 600, margin: '0 auto' }}>
            <BatchPanel onViewResult={handleViewBatchResult} />
          </div>
        ) : (
        <>
        {/* Left: Canvas area */}
        <div className="app-canvas-area">
          {/* Upload bar */}
          <div className="upload-bar">
            <input
              ref={fileInputRef}
              type="file"
              accept="image/jpeg,image/png,image/webp"
              onChange={handleFileChange}
              style={{ display: 'none' }}
            />
            <button
              className="upload-btn"
              onClick={() => fileInputRef.current?.click()}
              disabled={isLoading}
            >
              选择图像
            </button>
            {imageFile && (
              <span className="upload-filename">{imageFile.name}</span>
            )}
            {uploadError && (
              <span className="upload-error">{uploadError}</span>
            )}
          </div>

          <ImageCanvas
            image={imageEl}
            masks={coloredMasks}
            points={points}
            boxes={boxes}
            mode={mode}
            maskOpacity={maskOpacity}
            showMasks={showMasks}
            hoveredMaskIndex={hoveredMaskIndex}
            onPointAdd={handlePointAdd}
            onBoxComplete={handleBoxComplete}
            onMaskHover={handleMaskHover}
          />

          {/* Hovered mask tooltip */}
          {hoveredMaskIndex !== null && coloredMasks[hoveredMaskIndex] && (
            <div className="mask-tooltip">
              {coloredMasks[hoveredMaskIndex].label ?? `对象 ${hoveredMaskIndex + 1}`}
              {' — '}
              {(coloredMasks[hoveredMaskIndex].score * 100).toFixed(1)}%
            </div>
          )}
        </div>

        {/* Right: Controls */}
        <div className="app-sidebar">
          <PromptPanel
            mode={mode}
            textPrompt={textPrompt}
            points={points}
            boxes={boxes}
            isLoading={isLoading || !canSegment}
            onModeChange={setMode}
            onTextChange={setTextPrompt}
            onSubmit={handleSubmit}
            onClearPoints={handleClearPoints}
            onClearBoxes={handleClearBoxes}
          />

          {error && (
            <ErrorBanner
              message={error}
              onDismiss={() => clearResult()}
            />
          )}

          {/* "No matching objects" message (Requirements: 2.5, 7.4) */}
          {result && result.masks.length === 0 && !error && (
            <ErrorBanner message="未找到匹配对象" />
          )}

          <ResultPanel
            result={result ? { ...result, masks: coloredMasks } : null}
            maskOpacity={maskOpacity}
            showMasks={showMasks}
            onOpacityChange={handleOpacityChange}
            onToggleMasks={handleToggleMasks}
            onExportMask={handleExportMask}
            onExportOverlay={handleExportOverlay}
            onExportJSON={handleExportJSON}
          />
        </div>
        </>
        )}
      </div>
    </div>
  );
}

export default App;
