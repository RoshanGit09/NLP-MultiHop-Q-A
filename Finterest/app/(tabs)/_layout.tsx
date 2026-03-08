/**
 * app/(tabs)/_layout.tsx
 * This file is kept to satisfy Expo Router's file-based routing.
 * Navigation is fully handled by src/navigation/AppNavigator.tsx.
 * This component will never be rendered since _layout.tsx delegates to App.tsx.
 */

import React from 'react';
import { View } from 'react-native';

export default function TabLayout() {
  return <View />;
}
