const ALLOWED_EXTENSIONS = ['jpg', 'jpeg', 'png', 'webp'];

/**
 * Validate that a filename has an allowed image extension.
 * Accepts: jpg, jpeg, png, webp (case-insensitive).
 */
export function validateFileExtension(filename: string): boolean {
  const dotIndex = filename.lastIndexOf('.');
  if (dotIndex === -1 || dotIndex === filename.length - 1) {
    return false;
  }
  const ext = filename.slice(dotIndex + 1).toLowerCase();
  return ALLOWED_EXTENSIONS.includes(ext);
}
