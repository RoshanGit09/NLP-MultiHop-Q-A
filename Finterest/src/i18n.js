/**
 * i18n.js
 * Internationalization configuration using i18next + react-i18next
 * Supports: English, Tamil, Hindi, Malayalam, Telugu, Marathi
 * Persists selected language using AsyncStorage
 */

import AsyncStorage from '@react-native-async-storage/async-storage';
import * as Localization from 'expo-localization';
import i18n from 'i18next';
import { initReactI18next } from 'react-i18next';

// Import translation files
import en from './locales/en.json';
import hi from './locales/hi.json';
import ml from './locales/ml.json';
import mr from './locales/mr.json';
import ta from './locales/ta.json';
import te from './locales/te.json';

// Supported language codes
export const SUPPORTED_LANGUAGES = ['en', 'ta', 'hi', 'ml', 'te', 'mr'];

// AsyncStorage key for persisting language
const LANGUAGE_STORAGE_KEY = '@fintrace_language';

/**
 * Detects device locale and returns a supported language code.
 * Falls back to 'en' if the device language is not supported.
 */
const detectDeviceLanguage = () => {
  try {
    // Get device locale (e.g., "en-US", "hi-IN", "ta-IN")
    const deviceLocale = Localization.getLocales()[0]?.languageCode ?? 'en';
    const langCode = deviceLocale.split('-')[0].toLowerCase();
    return SUPPORTED_LANGUAGES.includes(langCode) ? langCode : 'en';
  } catch {
    return 'en';
  }
};

/**
 * Loads the saved language from AsyncStorage.
 * Returns null if no language has been saved yet.
 */
export const loadSavedLanguage = async () => {
  try {
    return await AsyncStorage.getItem(LANGUAGE_STORAGE_KEY);
  } catch {
    return null;
  }
};

/**
 * Persists the selected language to AsyncStorage.
 */
export const saveLanguage = async (lang) => {
  try {
    await AsyncStorage.setItem(LANGUAGE_STORAGE_KEY, lang);
  } catch (e) {
    console.warn('[i18n] Failed to save language:', e);
  }
};

/**
 * Initializes i18next with all translation resources.
 * Uses saved language or auto-detected device language.
 */
export const initI18n = async () => {
  // Determine initial language: saved > device detected > English
  const savedLang = await loadSavedLanguage();
  const deviceLang = detectDeviceLanguage();
  const initialLanguage = savedLang || deviceLang;

  await i18n.use(initReactI18next).init({
    compatibilityJSON: 'v3',
    resources: {
      en: { translation: en },
      ta: { translation: ta },
      hi: { translation: hi },
      ml: { translation: ml },
      te: { translation: te },
      mr: { translation: mr },
    },
    lng: initialLanguage,
    fallbackLng: 'en',
    interpolation: {
      escapeValue: false, // React already escapes values
    },
    react: {
      useSuspense: false, // Disable suspense for React Native
    },
  });

  return i18n;
};

export default i18n;
