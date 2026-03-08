/**
 * screens/LoginScreen.tsx
 * Firebase email/password login with multilingual support.
 * Includes language selector at the top.
 */

import { NativeStackNavigationProp } from '@react-navigation/native-stack';
import React, { useState } from 'react';
import { useTranslation } from 'react-i18next';
import {
    Alert,
    KeyboardAvoidingView,
    Platform,
    ScrollView,
    StatusBar,
    StyleSheet,
    Text,
    TouchableOpacity,
    View,
} from 'react-native';
import AppButton from '../components/AppButton';
import AppInput from '../components/AppInput';
import LanguageSelector from '../components/LanguageSelector';
import { COLORS, FONTS, RADIUS, SPACING } from '../constants/theme';
import { loginUser } from '../firebase/authService';
import { saveLanguage } from '../i18n';

type Props = {
  navigation: NativeStackNavigationProp<any>;
};

const LoginScreen: React.FC<Props> = ({ navigation }) => {
  const { t, i18n } = useTranslation();

  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [loading, setLoading] = useState(false);
  const [errors, setErrors] = useState<{ email?: string; password?: string }>({});

  // Validate form inputs
  const validate = (): boolean => {
    const newErrors: typeof errors = {};
    if (!email.trim() || !/\S+@\S+\.\S+/.test(email)) {
      newErrors.email = t('errors.invalidEmail');
    }
    if (!password || password.length < 6) {
      newErrors.password = t('errors.weakPassword');
    }
    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  // Map Supabase error messages to user-friendly text
  const getLoginError = (error: any): string => {
    const message: string = error?.message?.toLowerCase() || '';
    if (message.includes('invalid login credentials') || message.includes('invalid credentials') || message.includes('wrong password') || message.includes('user not found')) {
      return 'Incorrect email or password. Please try again.';
    }
    if (message.includes('email not confirmed')) {
      return 'Please confirm your email before logging in.';
    }
    if (message.includes('too many requests') || message.includes('rate limit')) {
      return 'Too many attempts. Please wait a few minutes and try again.';
    }
    if (message.includes('network') || message.includes('fetch')) {
      return t('errors.networkError');
    }
    return error?.message || t('errors.genericError');
  };

  const handleLogin = async () => {
    if (!validate()) return;
    setLoading(true);
    try {
      await loginUser(email.trim(), password);
      // Navigation to Home is handled automatically by AuthContext state change
    } catch (error: any) {
      Alert.alert(t('common.error'), getLoginError(error));
    } finally {
      setLoading(false);
    }
  };

  const handleLanguageChange = async (code: string) => {
    await i18n.changeLanguage(code);
    await saveLanguage(code);
  };

  return (
    <KeyboardAvoidingView
      style={styles.flex}
      behavior={Platform.OS === 'ios' ? 'padding' : undefined}
    >
      <StatusBar barStyle="light-content" backgroundColor={COLORS.primary} />
      <ScrollView
        contentContainerStyle={styles.container}
        keyboardShouldPersistTaps="handled"
        showsVerticalScrollIndicator={false}
      >
        {/* Language selector row */}
        <View style={styles.langRow}>
          <LanguageSelector
            value={i18n.language}
            onChange={handleLanguageChange}
            applyGlobally={false}
          />
        </View>

        {/* Header */}
        <View style={styles.header}>
          <View style={styles.logoContainer}>
            <Text style={styles.logoText}>FT</Text>
          </View>
          <Text style={styles.appName}>{t('app.name')}</Text>
          <Text style={styles.title}>{t('auth.loginTitle')}</Text>
          <Text style={styles.subtitle}>{t('auth.loginSubtitle')}</Text>
        </View>

        {/* Form */}
        <View style={styles.form}>
          <AppInput
            label={t('auth.email')}
            placeholder="you@example.com"
            value={email}
            onChangeText={setEmail}
            keyboardType="email-address"
            autoComplete="email"
            error={errors.email}
          />

          <AppInput
            label={t('auth.password')}
            placeholder="••••••••"
            value={password}
            onChangeText={setPassword}
            isPassword
            error={errors.password}
          />

          <TouchableOpacity style={styles.forgotRow}>
            <Text style={styles.forgotText}>{t('auth.forgotPassword')}</Text>
          </TouchableOpacity>

          <AppButton
            title={loading ? t('auth.loading') : t('auth.login')}
            onPress={handleLogin}
            loading={loading}
            style={styles.loginButton}
          />
        </View>

        {/* Signup link */}
        <TouchableOpacity
          style={styles.switchRow}
          onPress={() => navigation.navigate('Signup')}
        >
          <Text style={styles.switchText}>{t('auth.noAccount')}</Text>
        </TouchableOpacity>
      </ScrollView>
    </KeyboardAvoidingView>
  );
};

const styles = StyleSheet.create({
  flex: { flex: 1, backgroundColor: COLORS.background },
  container: {
    flexGrow: 1,
    paddingHorizontal: SPACING.xl,
    paddingTop: SPACING.xl,
    paddingBottom: SPACING.xxxl,
  },
  langRow: {
    alignItems: 'flex-end',
    marginBottom: SPACING.xl,
  },
  header: {
    alignItems: 'center',
    marginBottom: SPACING.xxl,
  },
  logoContainer: {
    width: 72,
    height: 72,
    borderRadius: RADIUS.lg,
    backgroundColor: COLORS.primary,
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: SPACING.md,
  },
  logoText: {
    fontSize: FONTS.xxl,
    fontWeight: '800',
    color: COLORS.white,
    letterSpacing: 1,
  },
  appName: {
    fontSize: FONTS.sm,
    fontWeight: '600',
    color: COLORS.accent,
    letterSpacing: 2,
    textTransform: 'uppercase',
    marginBottom: SPACING.sm,
  },
  title: {
    fontSize: FONTS.xxxl,
    fontWeight: '800',
    color: COLORS.textPrimary,
    marginBottom: SPACING.xs,
  },
  subtitle: {
    fontSize: FONTS.md,
    color: COLORS.textSecondary,
  },
  form: {
    gap: SPACING.xs,
  },
  forgotRow: {
    alignSelf: 'flex-end',
    marginTop: -SPACING.xs,
    marginBottom: SPACING.md,
  },
  forgotText: {
    fontSize: FONTS.sm,
    color: COLORS.accent,
    fontWeight: '600',
  },
  loginButton: {
    marginTop: SPACING.sm,
  },
  switchRow: {
    alignItems: 'center',
    marginTop: SPACING.xl,
  },
  switchText: {
    fontSize: FONTS.sm,
    color: COLORS.textSecondary,
  },
});

export default LoginScreen;
