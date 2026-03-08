/**
 * navigation/AppNavigator.tsx
 * Auth-gated navigator rendered inside Expo Router's NavigationContainer.
 * Does NOT wrap in its own NavigationContainer — Expo Router provides one.
 * Uses a bare Stack.Navigator so routing is disconnected from Expo Router's
 * file-based routes (React Navigation v7 allows nested independent stacks
 * when the component is rendered as a full-screen leaf in Expo Router).
 */

import React from 'react';
import { ActivityIndicator, StyleSheet, View } from 'react-native';
// NavigationIndependentTree is exported from @react-navigation/core v7.
// It creates a fully independent navigation tree so our Stack.Navigator
// does not conflict with Expo Router's NavigationContainer.
import { NavigationIndependentTree } from '@react-navigation/core';
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import { COLORS } from '../constants/theme';
import { useAuth } from '../context/AuthContext';
import AuthNavigator from './AuthNavigator';
import MainNavigator from './MainNavigator';

const Stack = createNativeStackNavigator();

const AppNavigator: React.FC = () => {
  const { user, isLoading } = useAuth();

  if (isLoading) {
    return (
      <View style={styles.loadingContainer}>
        <ActivityIndicator size="large" color={COLORS.accent} />
      </View>
    );
  }

  // NavigationIndependentTree wraps our own navigation tree completely
  // independently from Expo Router's tree — no nested container conflict.
  return (
    <NavigationIndependentTree>
      <Stack.Navigator screenOptions={{ headerShown: false }}>
        {user ? (
          <Stack.Screen name="Main" component={MainNavigator} />
        ) : (
          <Stack.Screen name="Auth" component={AuthNavigator} />
        )}
      </Stack.Navigator>
    </NavigationIndependentTree>
  );
};

const styles = StyleSheet.create({
  loadingContainer: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: COLORS.background,
  },
});

export default AppNavigator;
