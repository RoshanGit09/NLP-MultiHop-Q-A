/**
 * constants/theme.ts
 * Central design tokens for FinTraceQA.
 * Primary: #0A1F44 (deep navy)
 * Accent:  #1E90FF (electric blue)
 */

export const COLORS = {
  // Brand colors
  primary: '#0A1F44',
  primaryLight: '#1A3A6E',
  accent: '#1E90FF',
  accentLight: '#E8F4FF',

  // Backgrounds
  background: '#FFFFFF',
  surface: '#F5F7FA',
  card: '#FFFFFF',

  // Text
  textPrimary: '#0A1F44',
  textSecondary: '#6B7A99',
  textMuted: '#A0AABF',
  textOnPrimary: '#FFFFFF',

  // Status
  success: '#22C55E',
  error: '#EF4444',
  warning: '#F59E0B',
  info: '#3B82F6',

  // Neutrals
  white: '#FFFFFF',
  border: '#E2E8F0',
  divider: '#EEF2F7',
  shadow: 'rgba(10, 31, 68, 0.1)',

  // Chat
  userBubble: '#1E90FF',
  botBubble: '#F0F4FF',
  userBubbleText: '#FFFFFF',
  botBubbleText: '#0A1F44',
};

export const FONTS = {
  xs: 11,
  sm: 13,
  md: 15,
  lg: 17,
  xl: 20,
  xxl: 24,
  xxxl: 30,
};

export const SPACING = {
  xs: 4,
  sm: 8,
  md: 12,
  lg: 16,
  xl: 24,
  xxl: 32,
  xxxl: 48,
};

export const RADIUS = {
  sm: 8,
  md: 12,
  lg: 16,
  xl: 24,
  full: 999,
};

export const SHADOWS = {
  small: {
    shadowColor: COLORS.shadow,
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 1,
    shadowRadius: 4,
    elevation: 2,
  },
  medium: {
    shadowColor: COLORS.shadow,
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 1,
    shadowRadius: 8,
    elevation: 4,
  },
  large: {
    shadowColor: COLORS.shadow,
    shadowOffset: { width: 0, height: 8 },
    shadowOpacity: 1,
    shadowRadius: 16,
    elevation: 8,
  },
};
