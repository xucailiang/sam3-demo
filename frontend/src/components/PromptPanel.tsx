import { useCallback } from 'react';
import type { InteractionMode, PointPrompt, BoxPrompt } from '../types';
import { formatBoxCoordinates } from '../utils/boxDisplay';

export interface PromptPanelProps {
  mode: InteractionMode;
  textPrompt: string;
  points: PointPrompt[];
  boxes: BoxPrompt[];
  isLoading: boolean;
  onModeChange: (mode: InteractionMode) => void;
  onTextChange: (text: string) => void;
  onSubmit: () => void;
  onClearPoints: () => void;
  onClearBoxes: () => void;
}

const MODES: { value: InteractionMode; label: string }[] = [
  { value: 'text', label: '文本提示' },
  { value: 'point', label: '点击提示' },
  { value: 'box', label: '边界框' },
];

/**
 * PromptPanel provides mode switching (text / point / box),
 * text input, and clear buttons for points and boxes.
 *
 * Requirements: 2.1, 3.5, 4.4, 7.1, 7.3, 7.4
 */
export function PromptPanel({
  mode,
  textPrompt,
  points,
  boxes,
  isLoading,
  onModeChange,
  onTextChange,
  onSubmit,
  onClearPoints,
  onClearBoxes,
}: PromptPanelProps) {
  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent<HTMLInputElement>) => {
      if (e.key === 'Enter' && !isLoading) {
        onSubmit();
      }
    },
    [isLoading, onSubmit],
  );

  return (
    <div className="prompt-panel" style={{ padding: '1rem', display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
      {/* Mode switcher */}
      <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
        {MODES.map((m) => (
          <button
            key={m.value}
            onClick={() => onModeChange(m.value)}
            disabled={isLoading}
            style={{
              padding: '0.4rem 0.8rem',
              borderRadius: '4px',
              border: mode === m.value ? '2px solid #4ECDC4' : '1px solid #555',
              background: mode === m.value ? '#4ECDC4' : 'transparent',
              color: mode === m.value ? '#000' : 'inherit',
              cursor: isLoading ? 'not-allowed' : 'pointer',
              fontWeight: mode === m.value ? 600 : 400,
            }}
          >
            {m.label}
          </button>
        ))}
      </div>

      {/* Text mode input */}
      {mode === 'text' && (
        <div style={{ display: 'flex', gap: '0.5rem' }}>
          <input
            type="text"
            value={textPrompt}
            onChange={(e) => onTextChange(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="输入文本提示，多个用逗号分隔"
            disabled={isLoading}
            style={{
              flex: 1,
              padding: '0.5rem',
              borderRadius: '4px',
              border: '1px solid #555',
              background: '#333',
              color: '#fff',
            }}
          />
          <button
            onClick={onSubmit}
            disabled={isLoading || !textPrompt.trim()}
            style={{
              padding: '0.5rem 1rem',
              borderRadius: '4px',
              border: 'none',
              background: isLoading || !textPrompt.trim() ? '#555' : '#4ECDC4',
              color: '#000',
              cursor: isLoading || !textPrompt.trim() ? 'not-allowed' : 'pointer',
              fontWeight: 600,
            }}
          >
            {isLoading ? '处理中...' : '分割'}
          </button>
        </div>
      )}

      {/* Point mode controls */}
      {mode === 'point' && (
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', flexWrap: 'wrap' }}>
          <span style={{ fontSize: '0.9rem', color: '#aaa' }}>
            左键: 前景点 | 右键: 背景点
          </span>
          <span style={{ fontSize: '0.85rem' }}>
            已标记 {points.length} 个点
          </span>
          <button
            onClick={onClearPoints}
            disabled={isLoading || points.length === 0}
            style={{
              padding: '0.35rem 0.7rem',
              borderRadius: '4px',
              border: '1px solid #555',
              background: 'transparent',
              color: points.length === 0 ? '#555' : '#FF6B6B',
              cursor: isLoading || points.length === 0 ? 'not-allowed' : 'pointer',
            }}
          >
            清除点
          </button>
          <button
            onClick={onSubmit}
            disabled={isLoading || points.length === 0}
            style={{
              padding: '0.35rem 0.7rem',
              borderRadius: '4px',
              border: 'none',
              background: isLoading || points.length === 0 ? '#555' : '#4ECDC4',
              color: '#000',
              cursor: isLoading || points.length === 0 ? 'not-allowed' : 'pointer',
              fontWeight: 600,
            }}
          >
            {isLoading ? '处理中...' : '分割'}
          </button>
        </div>
      )}

      {/* Box mode controls */}
      {mode === 'box' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', flexWrap: 'wrap' }}>
            <span style={{ fontSize: '0.9rem', color: '#aaa' }}>
              拖拽绘制边界框
            </span>
            <span style={{ fontSize: '0.85rem' }}>
              已绘制 {boxes.length} 个框
            </span>
            <button
              onClick={onClearBoxes}
              disabled={isLoading || boxes.length === 0}
              style={{
                padding: '0.35rem 0.7rem',
                borderRadius: '4px',
                border: '1px solid #555',
                background: 'transparent',
                color: boxes.length === 0 ? '#555' : '#FF6B6B',
                cursor: isLoading || boxes.length === 0 ? 'not-allowed' : 'pointer',
              }}
            >
              清除框
            </button>
            <button
              onClick={onSubmit}
              disabled={isLoading || boxes.length === 0}
              style={{
                padding: '0.35rem 0.7rem',
                borderRadius: '4px',
                border: 'none',
                background: isLoading || boxes.length === 0 ? '#555' : '#4ECDC4',
                color: '#000',
                cursor: isLoading || boxes.length === 0 ? 'not-allowed' : 'pointer',
                fontWeight: 600,
              }}
            >
              {isLoading ? '处理中...' : '分割'}
            </button>
          </div>
          {boxes.length > 0 && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem' }}>
              {boxes.map((box, i) => {
                const coords = formatBoxCoordinates(box);
                return (
                  <span key={i} data-testid={`box-coords-${i}`} style={{ fontSize: '0.8rem', color: '#ccc' }}>
                    框 {i + 1}: {coords.display}
                  </span>
                );
              })}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
