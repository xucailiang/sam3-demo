import { useCallback } from 'react';

export interface ErrorBannerProps {
  /** Error message to display. If null/undefined, the banner is hidden. */
  message: string | null;
  /** Optional callback to dismiss the error. */
  onDismiss?: () => void;
}

/**
 * ErrorBanner displays a dismissible error message.
 *
 * Used for network errors, API errors, file format errors, and
 * "no matching objects" messages.
 *
 * Requirements: 1.4, 2.5, 7.4
 */
export function ErrorBanner({ message, onDismiss }: ErrorBannerProps) {
  const handleDismiss = useCallback(() => {
    onDismiss?.();
  }, [onDismiss]);

  if (!message) return null;

  return (
    <div
      role="alert"
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: '0.5rem',
        padding: '0.6rem 1rem',
        fontSize: '0.85rem',
        color: '#FF6B6B',
        background: 'rgba(255, 107, 107, 0.08)',
        borderRadius: '4px',
        border: '1px solid rgba(255, 107, 107, 0.25)',
      }}
    >
      <span style={{ flex: 1 }}>{message}</span>
      {onDismiss && (
        <button
          onClick={handleDismiss}
          aria-label="关闭错误提示"
          style={{
            background: 'none',
            border: 'none',
            color: '#FF6B6B',
            cursor: 'pointer',
            fontSize: '1rem',
            lineHeight: 1,
            padding: '0 0.25rem',
          }}
        >
          ×
        </button>
      )}
    </div>
  );
}
