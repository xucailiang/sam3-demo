import type { FilterStats } from '../utils/filter';

export interface FilterControlsProps {
  /** Current confidence threshold (0-1) */
  threshold: number;
  /** Threshold change callback */
  onThresholdChange: (threshold: number) => void;
  /** Filter statistics */
  stats: FilterStats;
  /** Select all callback */
  onSelectAll: () => void;
  /** Select none callback */
  onSelectNone: () => void;
  /** Invert selection callback */
  onInvertSelection: () => void;
  /** Whether controls are disabled */
  disabled?: boolean;
}

/**
 * FilterControls provides confidence threshold slider and batch selection buttons.
 *
 * Requirements: 1.1, 1.3, 4.1, 4.2, 4.3, 5.1, 5.2
 */
export function FilterControls({
  threshold,
  onThresholdChange,
  stats,
  onSelectAll,
  onSelectNone,
  onInvertSelection,
  disabled = false,
}: FilterControlsProps) {
  const handleSliderChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    // Convert from percentage (0-100) to decimal (0-1)
    const value = Number(e.target.value) / 100;
    onThresholdChange(value);
  };

  const thresholdPercent = Math.round(threshold * 100);

  return (
    <div style={containerStyle}>
      {/* Confidence threshold slider */}
      <div style={sliderContainerStyle}>
        <label htmlFor="confidence-threshold" style={labelStyle}>
          置信度阈值
        </label>
        <input
          id="confidence-threshold"
          type="range"
          min={0}
          max={100}
          step={1}
          value={thresholdPercent}
          onChange={handleSliderChange}
          disabled={disabled}
          style={sliderStyle}
        />
        <span style={valueStyle}>{thresholdPercent}%</span>
      </div>

      {/* Statistics display */}
      <div style={statsStyle}>
        <span>检测 {stats.totalCount} 个</span>
        <span style={statsSeparator}>·</span>
        <span>过滤后 {stats.filteredCount} 个</span>
        <span style={statsSeparator}>·</span>
        <span style={selectedStyle}>已选 {stats.selectedCount} 个</span>
      </div>

      {/* Batch operation buttons */}
      <div style={buttonsContainerStyle}>
        <button
          onClick={onSelectAll}
          disabled={disabled || stats.filteredCount === 0}
          style={disabled || stats.filteredCount === 0 ? btnDisabled : btnSecondary}
        >
          全选
        </button>
        <button
          onClick={onSelectNone}
          disabled={disabled || stats.selectedCount === 0}
          style={disabled || stats.selectedCount === 0 ? btnDisabled : btnSecondary}
        >
          全不选
        </button>
        <button
          onClick={onInvertSelection}
          disabled={disabled || stats.filteredCount === 0}
          style={disabled || stats.filteredCount === 0 ? btnDisabled : btnSecondary}
        >
          反选
        </button>
      </div>
    </div>
  );
}

// --- Styles ---
const containerStyle: React.CSSProperties = {
  display: 'flex',
  flexDirection: 'column',
  gap: '0.5rem',
  padding: '0.5rem 0',
  borderBottom: '1px solid #444',
  marginBottom: '0.5rem',
};

const sliderContainerStyle: React.CSSProperties = {
  display: 'flex',
  alignItems: 'center',
  gap: '0.5rem',
};

const labelStyle: React.CSSProperties = {
  fontSize: '0.85rem',
  color: '#ccc',
  whiteSpace: 'nowrap',
};

const sliderStyle: React.CSSProperties = {
  flex: 1,
  cursor: 'pointer',
};

const valueStyle: React.CSSProperties = {
  fontSize: '0.85rem',
  minWidth: '3rem',
  textAlign: 'right',
  color: '#4ECDC4',
  fontWeight: 600,
};

const statsStyle: React.CSSProperties = {
  display: 'flex',
  alignItems: 'center',
  gap: '0.25rem',
  fontSize: '0.8rem',
  color: '#aaa',
  flexWrap: 'wrap',
};

const statsSeparator: React.CSSProperties = {
  color: '#555',
};

const selectedStyle: React.CSSProperties = {
  color: '#4ECDC4',
};

const buttonsContainerStyle: React.CSSProperties = {
  display: 'flex',
  gap: '0.5rem',
  flexWrap: 'wrap',
};

const btnSecondary: React.CSSProperties = {
  padding: '0.3rem 0.6rem',
  borderRadius: 4,
  border: '1px solid #555',
  background: 'transparent',
  color: 'inherit',
  cursor: 'pointer',
  fontSize: '0.8rem',
};

const btnDisabled: React.CSSProperties = {
  ...btnSecondary,
  color: '#555',
  cursor: 'not-allowed',
};
