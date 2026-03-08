/**
 * screens/SignupScreen.tsx
 * New user registration with full profile fields.
 * Collects: name, email, password, language, investor_type, risk_appetite.
 * Creates Supabase Auth account + profiles table record.
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
import { signUpUser } from '../firebase/authService';

type Props = {
  navigation: NativeStackNavigationProp<any>;
};

// Option pill component for selecting investor type / risk appetite
const OptionPill: React.FC<{
  label: string;
  selected: boolean;
  onPress: () => void;
}> = ({ label, selected, onPress }) => (
  <TouchableOpacity
    style={[styles.pill, selected && styles.pillSelected]}
    onPress={onPress}
    activeOpacity={0.8}
  >
    <Text style={[styles.pillText, selected && styles.pillTextSelected]}>{label}</Text>
  </TouchableOpacity>
);

const SignupScreen: React.FC<Props> = ({ navigation }) => {
  const { t, i18n } = useTranslation();

  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [language, setLanguage] = useState(i18n.language || 'en');
  const [investorType, setInvestorType] = useState<'Retail' | 'Institutional' | 'Student'>('Retail');
  const [riskAppetite, setRiskAppetite] = useState<'Low' | 'Medium' | 'High'>('Medium');
  const [loading, setLoading] = useState(false);
  const [errors, setErrors] = useState<Record<string, string>>({});

  // Validate all form fields
  const validate = (): boolean => {
    const newErrors: Record<string, string> = {};
    if (!name.trim() || name.trim().length < 2) newErrors.name = t('errors.nameTooShort');
    if (!email.trim() || !/\S+@\S+\.\S+/.test(email)) newErrors.email = t('errors.invalidEmail');
    if (!password || password.length < 6) newErrors.password = t('errors.weakPassword');
    if (password !== confirmPassword) newErrors.confirmPassword = t('errors.passwordMismatch');
    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const getSignupError = (error: any): string => {
    const message: string = error?.message?.toLowerCase() || '';
    if (message.includes('rate limit') || message.includes('too many')) {
      return 'Too many signup attempts. Please wait a few minutes and try again.';
    }
    if (message.includes('already registered') || message.includes('already in use') || message.includes('user already')) {
      return 'This email is already registered. Please log in instead.';
    }
    if (message.includes('invalid email')) return t('errors.invalidEmail');
    if (message.includes('network') || message.includes('fetch')) return t('errors.networkError');
    if (message.includes('weak password') || message.includes('password')) return t('errors.weakPassword');
    return error?.message || t('errors.genericError');
  };

  const handleSignup = async () => {
    if (!validate()) return;
    setLoading(true);
    try {
      console.log('[SignupScreen] Starting signup with data:', {
        name: name.trim(),
        email: email.trim(),
        language,
        investorType,
        riskAppetite,
      });
      
      await signUpUser(email.trim(), password, {
        name: name.trim(),
        email: email.trim(),
        language,
        investor_type: investorType,
        risk_appetite: riskAppetite,
      });
      
      console.log('[SignupScreen] Signup successful!');
      Alert.alert(
        t('common.success'),
        'Account created successfully! Please check your profile.',
        [{ text: 'OK' }]
      );
      // Auth context will pick up the new user automatically
    } catch (error: any) {
      console.error('[SignupScreen] Signup error:', error);
      const errorMessage = getSignupError(error);
      Alert.alert(t('common.error'), errorMessage);
    } finally {
      setLoading(false);
    }
  };

  const investorOptions: Array<{ value: 'Retail' | 'Institutional' | 'Student'; label: string }> = [
    { value: 'Retail', label: t('auth.retail') },
    { value: 'Institutional', label: t('auth.institutional') },
    { value: 'Student', label: t('auth.student') },
  ];

  const riskOptions: Array<{ value: 'Low' | 'Medium' | 'High'; label: string }> = [
    { value: 'Low', label: t('auth.low') },
    { value: 'Medium', label: t('auth.medium') },
    { value: 'High', label: t('auth.high') },
  ];

  return (
    <KeyboardAvoidingView
      style={styles.flex}
      behavior={Platform.OS === 'ios' ? 'padding' : undefined}
    >
      <StatusBar barStyle="dark-content" backgroundColor={COLORS.background} />
      <ScrollView
        contentContainerStyle={styles.container}
        keyboardShouldPersistTaps="handled"
        showsVerticalScrollIndicator={false}
      >
        {/* Header */}
        <View style={styles.header}>
          <TouchableOpacity onPress={() => navigation.goBack()} style={styles.backButton}>
            <Text style={styles.backText}>← {t('common.back')}</Text>
          </TouchableOpacity>
          <Text style={styles.title}>{t('auth.signupTitle')}</Text>
          <Text style={styles.subtitle}>{t('auth.signupSubtitle')}</Text>
        </View>

        {/* Form fields */}
        <AppInput
          label={t('auth.fullName')}
          placeholder="John Doe"
          value={name}
          onChangeText={setName}
          autoCapitalize="words"
          error={errors.name}
        />

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

        <AppInput
          label={t('auth.confirmPassword')}
          placeholder="••••••••"
          value={confirmPassword}
          onChangeText={setConfirmPassword}
          isPassword
          error={errors.confirmPassword}
        />

        {/* Preferred Language */}
        <View style={styles.fieldGroup}>
          <Text style={styles.fieldLabel}>{t('auth.preferredLanguage')}</Text>
          <LanguageSelector value={language} onChange={setLanguage} />
        </View>

        {/* Investor Type */}
        <View style={styles.fieldGroup}>
          <Text style={styles.fieldLabel}>{t('auth.investorType')}</Text>
          <View style={styles.pillRow}>
            {investorOptions.map((opt) => (
              <OptionPill
                key={opt.value}
                label={opt.label}
                selected={investorType === opt.value}
                onPress={() => setInvestorType(opt.value)}
              />
            ))}
          </View>
        </View>

        {/* Risk Appetite */}
        <View style={styles.fieldGroup}>
          <Text style={styles.fieldLabel}>{t('auth.riskAppetite')}</Text>
          <View style={styles.pillRow}>
            {riskOptions.map((opt) => (
              <OptionPill
                key={opt.value}
                label={opt.label}
                selected={riskAppetite === opt.value}
                onPress={() => setRiskAppetite(opt.value)}
              />
            ))}
          </View>
        </View>

        {/* Submit */}
        <AppButton
          title={loading ? t('auth.loading') : t('auth.signup')}
          onPress={handleSignup}
          loading={loading}
          style={styles.signupButton}
        />

        {/* Login link */}
        <TouchableOpacity style={styles.switchRow} onPress={() => navigation.navigate('Login')}>
          <Text style={styles.switchText}>{t('auth.hasAccount')}</Text>
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
    paddingTop: SPACING.lg,
    paddingBottom: SPACING.xxxl,
  },
  header: {
    marginBottom: SPACING.xl,
  },
  backButton: {
    marginBottom: SPACING.lg,
  },
  backText: {
    fontSize: FONTS.md,
    color: COLORS.accent,
    fontWeight: '600',
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
  fieldGroup: {
    marginBottom: SPACING.md,
  },
  fieldLabel: {
    fontSize: FONTS.sm,
    fontWeight: '600',
    color: COLORS.textPrimary,
    marginBottom: SPACING.sm,
  },
  pillRow: {
    flexDirection: 'row',
    gap: SPACING.sm,
    flexWrap: 'wrap',
  },
  pill: {
    paddingHorizontal: SPACING.lg,
    paddingVertical: SPACING.sm,
    borderRadius: RADIUS.full,
    borderWidth: 1.5,
    borderColor: COLORS.border,
    backgroundColor: COLORS.surface,
  },
  pillSelected: {
    borderColor: COLORS.accent,
    backgroundColor: COLORS.accentLight,
  },
  pillText: {
    fontSize: FONTS.sm,
    color: COLORS.textSecondary,
    fontWeight: '500',
  },
  pillTextSelected: {
    color: COLORS.accent,
    fontWeight: '700',
  },
  signupButton: {
    marginTop: SPACING.lg,
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

export default SignupScreen;
