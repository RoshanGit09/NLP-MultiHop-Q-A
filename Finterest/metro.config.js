// metro.config.js
// Metro bundler configuration for FinTraceQA.
// Ensures JSON translation files in src/locales are resolved correctly.

const { getDefaultConfig } = require('expo/metro-config');

const config = getDefaultConfig(__dirname);

// Ensure JSON files are treated as assets for i18next translations
config.resolver.assetExts = config.resolver.assetExts.filter((ext) => ext !== 'json');
config.resolver.sourceExts = [...config.resolver.sourceExts, 'json'];

module.exports = config;
