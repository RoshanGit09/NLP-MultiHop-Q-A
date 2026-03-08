/**
 * app/_layout.tsx
 * Expo Router root layout for FinTraceQA.
 *
 * This wraps ALL screens with i18n + AuthProvider.
 * The actual navigation (Auth stack / Main tabs) lives in app/index.tsx
 * which Expo Router renders as the single index route inside this layout.
 * Expo Router's own Stack IS the NavigationContainer — we never create another.
 */

import { Slot } from 'expo-router';
import React, { useEffect, useState } from 'react';
import { ActivityIndicator, StyleSheet, View } from 'react-native';
import 'react-native-reanimated';
import { COLORS } from '../src/constants/theme';
import { AuthProvider } from '../src/context/AuthContext';
import { initI18n } from '../src/i18n';

export default function RootLayout() {
  const [i18nReady, setI18nReady] = useState(false);

  useEffect(() => {
    // Initialize i18next (loads persisted language) before first render
    initI18n().then(() => setI18nReady(true));
  }, []);

  if (!i18nReady) {
    // Splash while i18n loads
    return (
      <View style={styles.splash}>
        <ActivityIndicator size="large" color={COLORS.accent} />
      </View>
    );
  }

  // <Slot /> renders the matched child route (app/index.tsx).
  // Expo Router's NavigationContainer is already active — we must NOT add another.
  return (
    <AuthProvider>
      <Slot />
    </AuthProvider>
  );
}

const styles = StyleSheet.create({
  splash: {
    flex: 1,
    backgroundColor: COLORS.primary,
    alignItems: 'center',
    justifyContent: 'center',
  },
});
