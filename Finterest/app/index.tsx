/**
 * app/index.tsx
 * Single Expo Router entry point for FinTraceQA.
 *
 * Renders AppNavigator which contains:
 *   - AuthNavigator  (Login / Signup) — when user is not authenticated
 *   - MainNavigator  (News / Chatbot / Profile tabs) — when authenticated
 *
 * This component is rendered inside Expo Router's NavigationContainer via
 * the <Slot /> in _layout.tsx. AppNavigator uses a bare Stack.Navigator
 * (no NavigationContainer of its own) so there is no nesting conflict.
 */

import React from 'react';
import { StyleSheet, View } from 'react-native';
import { COLORS } from '../src/constants/theme';
import AppNavigator from '../src/navigation/AppNavigator';

export default function Index() {
  return (
    <View style={styles.container}>
      <AppNavigator />
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: COLORS.background,
  },
});
