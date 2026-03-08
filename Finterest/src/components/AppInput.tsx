/**
 * components/AppInput.tsx
 * Reusable text input component with label and error state.
 */

import { Ionicons } from '@expo/vector-icons';
import React, { useState } from 'react';
import {
    StyleSheet,
    Text,
    TextInput,
    TextInputProps,
    TouchableOpacity,
    View,
    ViewStyle,
} from 'react-native';
import { COLORS, FONTS, RADIUS, SPACING } from '../constants/theme';

interface AppInputProps extends TextInputProps {
  label?: string;
  error?: string;
  containerStyle?: ViewStyle;
  isPassword?: boolean;
}

const AppInput: React.FC<AppInputProps> = ({
  label,
  error,
  containerStyle,
  isPassword = false,
  ...textInputProps
}) => {
  const [showPassword, setShowPassword] = useState(false);

  return (
    <View style={[styles.container, containerStyle]}>
      {label ? <Text style={styles.label}>{label}</Text> : null}

      <View style={[styles.inputWrapper, error ? styles.inputError : styles.inputNormal]}>
        <TextInput
          style={styles.input}
          placeholderTextColor={COLORS.textMuted}
          secureTextEntry={isPassword && !showPassword}
          autoCapitalize="none"
          {...textInputProps}
        />

        {/* Show/hide password toggle */}
        {isPassword && (
          <TouchableOpacity
            onPress={() => setShowPassword((prev) => !prev)}
            style={styles.eyeButton}
          >
            <Ionicons
              name={showPassword ? 'eye-off-outline' : 'eye-outline'}
              size={20}
              color={COLORS.textSecondary}
            />
          </TouchableOpacity>
        )}
      </View>

      {/* Error message */}
      {error ? <Text style={styles.errorText}>{error}</Text> : null}
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    marginBottom: SPACING.md,
  },
  label: {
    fontSize: FONTS.sm,
    fontWeight: '600',
    color: COLORS.textPrimary,
    marginBottom: SPACING.xs,
  },
  inputWrapper: {
    flexDirection: 'row',
    alignItems: 'center',
    borderWidth: 1.5,
    borderRadius: RADIUS.md,
    backgroundColor: COLORS.surface,
    paddingHorizontal: SPACING.md,
  },
  inputNormal: {
    borderColor: COLORS.border,
  },
  inputError: {
    borderColor: COLORS.error,
  },
  input: {
    flex: 1,
    height: 50,
    fontSize: FONTS.md,
    color: COLORS.textPrimary,
  },
  eyeButton: {
    padding: SPACING.xs,
  },
  errorText: {
    fontSize: FONTS.xs,
    color: COLORS.error,
    marginTop: SPACING.xs,
  },
});

export default AppInput;
