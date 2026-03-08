/**
 * src/App.tsx
 * Root application component.
 * Initializes i18n, wraps app in AuthProvider, renders AppNavigator.
 */

import React, { useEffect, useState } from 'react';
import { ActivityIndicator, StyleSheet, View } from 'react-native';
import { COLORS } from './constants/theme';
import { AuthProvider } from './context/AuthContext';
import { initI18n } from './i18n';
import AppNavigator from './navigation/AppNavigator';

const App: React.FC = () => {
  const [i18nReady, setI18nReady] = useState(false);

  useEffect(() => {
    // Initialize i18next with saved/detected language before rendering UI
    initI18n().then(() => setI18nReady(true));
  }, []);

  // Wait for i18n to load before rendering
  if (!i18nReady) {
    return (
      <View style={styles.splash}>
        <ActivityIndicator size="large" color={COLORS.accent} />
      </View>
    );
  }

  return (
    <AuthProvider>
      <AppNavigator />
    </AuthProvider>
  );
};

const styles = StyleSheet.create({
  splash: {
    flex: 1,
    backgroundColor: COLORS.primary,
    alignItems: 'center',
    justifyContent: 'center',
  },
});

export default App;
