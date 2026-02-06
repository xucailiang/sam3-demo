import type { SegmentationResult } from '../types';

export interface ResultPanelProps {
  result: SegmentationResult | null;
  maskOpacity: number;
  showMasks: boolean;
  onOpacityChange: (opacity: number) => void;
  onToggleMasks: () => void;
  onExportMask: () => void;
  onExportOverlay: () => void;
  onExportJSON: () => void;
}

/**
 * ResultPanel displays detection statistics, mask opacity slider,
 * mask visibility toggle, and export buttons.
 *
 * Requirements: 5.3, 5.4, 5.5
 */
export function ResultPanel({
  result,
  maskOpacity,
  showMasks,
  onOpacityChange,
  onToggleMasks,
  onExportMask,
  onExportOverlay,
  onExportJSON,
}: ResultPanelProps) {
  if (!result) {
    return (
      <div className="result-panel" style={{ padding: '1rem', color: '#888' }}>
        暂无分割结果
      </div>
    );
  }

  return (
    <div className="result-panel" style={{ padding: '1rem', display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
      {/* Statistics */}
      <div style={{ display: 'flex', gap: '1.5rem', fontSize: '0.95rem' }}>
        <span>检测对象: <strong>{result.count}</strong></span>
        <span>耗时: <strong>{result.processingTimeMs.toFixed(0)} ms</strong></span>
      </div>

      {/* Mask list with labels and scores */}
      {result.masks.length > 0 && (
        <div style={{ fontSize: '0.85rem', display: 'flex', flexDirection: 'column', gap: '0.25rem' }}>
          {result.masks.map((m, i) => (
            <div key={i} style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <span
                style={{
                  display: 'inline-block',
                  width: 12,
                  height: 12,
                  borderRadius: 2,
                  background: m.color,
                }}
              />
              <span>{m.label ?? `对象 ${i + 1}`}</span>
              <span style={{ color: '#aaa' }}>
                {(m.score * 100).toFixed(1)}%
              </span>
            </div>
          ))}
        </div>
      )}

      {/* Visibility toggle */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
        <button
          onClick={onToggleMasks}
          style={{
            padding: '0.35rem 0.7rem',
            borderRadius: '4px',
            border: '1px solid #555',
            background: showMasks ? '#4ECDC4' : 'transparent',
            color: showMasks ? '#000' : 'inherit',
            cursor: 'pointer',
          }}
        >
          {showMasks ? '隐藏掩码' : '显示掩码'}
        </button>
      </div>

      {/* Opacity slider */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
        <label htmlFor="opacity-slider" style={{ fontSize: '0.9rem', whiteSpace: 'nowrap' }}>
          透明度
        </label>
        <input
          id="opacity-slider"
          type="range"
          min={0}
          max={1}
          step={0.05}
          value={maskOpacity}
          onChange={(e) => onOpacityChange(Number(e.target.value))}
          style={{ flex: 1 }}
        />
        <span style={{ fontSize: '0.85rem', minWidth: '2.5rem', textAlign: 'right' }}>
          {(maskOpacity * 100).toFixed(0)}%
        </span>
      </div>

      {/* Export buttons */}
      <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
        <button onClick={onExportMask} style={exportBtnStyle}>导出掩码</button>
        <button onClick={onExportOverlay} style={exportBtnStyle}>导出叠加图</button>
        <button onClick={onExportJSON} style={exportBtnStyle}>导出 JSON</button>
      </div>
    </div>
  );
}

const exportBtnStyle: React.CSSProperties = {
  padding: '0.35rem 0.7rem',
  borderRadius: '4px',
  border: '1px solid #555',
  background: 'transparent',
  color: 'inherit',
  cursor: 'pointer',
  fontSize: '0.85rem',
};
