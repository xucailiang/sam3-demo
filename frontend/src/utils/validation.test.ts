import { describe, it, expect } from 'vitest';
import { validateFileExtension } from './validation';

describe('validateFileExtension', () => {
  it('accepts valid image extensions', () => {
    expect(validateFileExtension('photo.jpg')).toBe(true);
    expect(validateFileExtension('photo.jpeg')).toBe(true);
    expect(validateFileExtension('photo.png')).toBe(true);
    expect(validateFileExtension('photo.webp')).toBe(true);
  });

  it('is case-insensitive', () => {
    expect(validateFileExtension('photo.JPG')).toBe(true);
    expect(validateFileExtension('photo.Png')).toBe(true);
    expect(validateFileExtension('photo.WEBP')).toBe(true);
  });

  it('rejects invalid extensions', () => {
    expect(validateFileExtension('photo.gif')).toBe(false);
    expect(validateFileExtension('photo.bmp')).toBe(false);
    expect(validateFileExtension('photo.svg')).toBe(false);
    expect(validateFileExtension('photo.txt')).toBe(false);
  });

  it('rejects files with no extension', () => {
    expect(validateFileExtension('photo')).toBe(false);
    expect(validateFileExtension('photo.')).toBe(false);
  });

  it('uses last extension for dotted filenames', () => {
    expect(validateFileExtension('my.photo.png')).toBe(true);
    expect(validateFileExtension('my.photo.txt')).toBe(false);
  });
});
