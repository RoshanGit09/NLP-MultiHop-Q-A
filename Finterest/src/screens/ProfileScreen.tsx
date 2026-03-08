/**
 * screens/ProfileScreen.tsx
 * User profile display with edit, language change, and logout actions.
 */

import { Ionicons } from '@expo/vector-icons';
import React, { useState } from 'react';
import { useTranslation } from 'react-i18next';
import {
    Alert,
    Modal,
    Platform,
    SafeAreaView,
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
import { COLORS, FONTS, RADIUS, SHADOWS, SPACING } from '../constants/theme';
import { useAuth } from '../context/AuthContext';
import { logoutUser, updateUserProfile } from '../firebase/authService';
import { saveLanguage } from '../i18n';

// Pill option for investor type / risk appetite
const OptionPill: React.FC<{ label: string; selected: boolean; onPress: () => void }> = ({
  label,
  selected,
  onPress,
}) => (
  <TouchableOpacity
    style={[pillStyles.pill, selected && pillStyles.pillSelected]}
    onPress={onPress}
    activeOpacity={0.8}
  >
    <Text style={[pillStyles.pillText, selected && pillStyles.pillTextSelected]}>{label}</Text>
  </TouchableOpacity>
);

// Profile info row component
const InfoRow: React.FC<{ icon: string; label: string; value: string }> = ({
  icon,
  label,
  value,
}) => (
  <View style={styles.infoRow}>
    <View style={styles.infoIconWrapper}>
      <Ionicons name={icon as any} size={18} color={COLORS.accent} />
    </View>
    <View style={styles.infoContent}>
      <Text style={styles.infoLabel}>{label}</Text>
      <Text style={styles.infoValue}>{value}</Text>
    </View>
  </View>
);

const ProfileScreen: React.FC = () => {
  const { t, i18n } = useTranslation();
  const { user, userProfile, setUserProfile } = useAuth();

  const [editModalVisible, setEditModalVisible] = useState(false);
  const [editName, setEditName] = useState(userProfile?.name || '');
  const [editLanguage, setEditLanguage] = useState(userProfile?.language || i18n.language);
  const [editInvestorType, setEditInvestorType] = useState<'Retail' | 'Institutional' | 'Student'>(
    userProfile?.investor_type || 'Retail'
  );
  const [editRisk, setEditRisk] = useState<'Low' | 'Medium' | 'High'>(
    userProfile?.risk_appetite || 'Medium'
  );
  const [saving, setSaving] = useState(false);
  const [loggingOut, setLoggingOut] = useState(false);

  // Format Firestore timestamp for display
  const formatDate = (ts: any): string => {
    if (!ts) return '—';
    try {
      const date = ts.toDate ? ts.toDate() : new Date(ts);
      return date.toLocaleDateString(undefined, { year: 'numeric', month: 'long', day: 'numeric' });
    } catch {
      return '—';
    }
  };

  // Handle language change globally
  const handleLanguageChange = async (code: string) => {
    await i18n.changeLanguage(code);
    await saveLanguage(code);
    // Also update Firestore profile
    if (user) {
      try {
        await updateUserProfile(user.id, { language: code });
        setUserProfile(userProfile ? { ...userProfile, language: code } : null);
      } catch {/* non-critical */}
    }
  };

  // Handle profile save
  const handleSaveProfile = async () => {
    if (!user) return;
    if (!editName.trim() || editName.trim().length < 2) {
      Alert.alert(t('common.error'), t('errors.nameTooShort'));
      return;
    }
    setSaving(true);
    try {
      await updateUserProfile(user.id, {
        name: editName.trim(),
        language: editLanguage,
        investor_type: editInvestorType,
        risk_appetite: editRisk,
      });
      setUserProfile({
        ...(userProfile || { id: user.id, email: user.email || '', created_at: new Date().toISOString() }),
        name: editName.trim(),
        language: editLanguage,
        investor_type: editInvestorType,
        risk_appetite: editRisk,
      });
      // Apply language change
      if (editLanguage !== i18n.language) {
        await i18n.changeLanguage(editLanguage);
        await saveLanguage(editLanguage);
      }
      setEditModalVisible(false);
      Alert.alert(t('common.success'), t('profile.updateSuccess'));
    } catch (err: any) {
      console.error('[ProfileScreen] Save error:', err);
      Alert.alert(t('common.error'), err?.message || t('errors.genericError'));
    } finally {
      setSaving(false);
    }
  };

  // Handle logout
  const handleLogout = async () => {
    Alert.alert(t('auth.logout'), t('profile.logout') + '?', [
      { text: t('common.cancel'), style: 'cancel' },
      {
        text: t('auth.logout'),
        style: 'destructive',
        onPress: async () => {
          setLoggingOut(true);
          try {
            await logoutUser();
          } catch {
            Alert.alert(t('common.error'), t('errors.genericError'));
          } finally {
            setLoggingOut(false);
          }
        },
      },
    ]);
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

  const languageLabel = t(`language.${userProfile?.language || i18n.language}`) || 'English';

  return (
    <SafeAreaView style={styles.safeArea}>
      <StatusBar barStyle="light-content" backgroundColor={COLORS.primary} />

      <ScrollView showsVerticalScrollIndicator={false}>
        {/* Setup warning banner if profile data is missing */}
        {user && !userProfile && (
          <View style={styles.setupBanner}>
            <Ionicons name="information-circle" size={24} color={COLORS.warning} />
            <View style={styles.setupBannerText}>
              <Text style={styles.setupBannerTitle}>Profile Data Missing</Text>
              <Text style={styles.setupBannerDesc}>
                Your profile wasn't created properly. Please log out and sign up again, or contact support.
              </Text>
            </View>
          </View>
        )}

        {/* Header banner */}
        <View style={styles.headerBanner}>
          <View style={styles.avatarCircle}>
            <Text style={styles.avatarText}>
              {(userProfile?.name || user?.email || 'U')[0].toUpperCase()}
            </Text>
          </View>
          <Text style={styles.profileName}>{userProfile?.name || '—'}</Text>
          <Text style={styles.profileEmail}>{user?.email || '—'}</Text>
          <View style={styles.badgeRow}>
            <View style={styles.badge}>
              <Text style={styles.badgeText}>{userProfile?.investor_type || '—'}</Text>
            </View>
            <View style={[styles.badge, styles.riskBadge]}>
              <Text style={styles.badgeText}>{userProfile?.risk_appetite || '—'} Risk</Text>
            </View>
          </View>
        </View>

        {/* Info section */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>{t('profile.title')}</Text>

          <View style={styles.infoCard}>
            <InfoRow icon="person-outline" label={t('profile.name')} value={userProfile?.name || '—'} />
            <View style={styles.divider} />
            <InfoRow icon="mail-outline" label={t('profile.email')} value={user?.email || '—'} />
            <View style={styles.divider} />
            <InfoRow icon="globe-outline" label={t('profile.language')} value={languageLabel} />
            <View style={styles.divider} />
            <InfoRow icon="briefcase-outline" label={t('profile.investorType')} value={userProfile?.investor_type || '—'} />
            <View style={styles.divider} />
            <InfoRow icon="trending-up-outline" label={t('profile.riskAppetite')} value={userProfile?.risk_appetite || '—'} />
            <View style={styles.divider} />
            <InfoRow
              icon="calendar-outline"
              label={t('profile.memberSince')}
              value={formatDate(userProfile?.created_at)}
            />
          </View>
        </View>

        {/* Actions section */}
        <View style={styles.section}>
          {/* Edit Profile */}
          <TouchableOpacity
            style={styles.actionButton}
            onPress={() => {
              setEditName(userProfile?.name || '');
              setEditLanguage(userProfile?.language || i18n.language);
              setEditInvestorType(userProfile?.investor_type || 'Retail');
              setEditRisk(userProfile?.risk_appetite || 'Medium');
              setEditModalVisible(true);
            }}
          >
            <Ionicons name="create-outline" size={22} color={COLORS.accent} />
            <Text style={styles.actionText}>{t('profile.editProfile')}</Text>
            <Ionicons name="chevron-forward" size={18} color={COLORS.textMuted} />
          </TouchableOpacity>

          {/* Change Language */}
          <View style={styles.actionButton}>
            <Ionicons name="language-outline" size={22} color={COLORS.accent} />
            <Text style={styles.actionText}>{t('profile.changeLanguage')}</Text>
            <LanguageSelector
              value={i18n.language}
              onChange={handleLanguageChange}
              applyGlobally={false}
            />
          </View>

          {/* Logout */}
          <AppButton
            title={loggingOut ? t('common.loading') : t('auth.logout')}
            onPress={handleLogout}
            variant="danger"
            loading={loggingOut}
            style={styles.logoutButton}
          />
        </View>
      </ScrollView>

      {/* Edit Profile Modal */}
      <Modal
        visible={editModalVisible}
        animationType="slide"
        transparent
        onRequestClose={() => setEditModalVisible(false)}
      >
        <View style={styles.modalOverlay}>
          <SafeAreaView style={styles.modalSheet}>
            <View style={styles.modalHeader}>
              <Text style={styles.modalTitle}>{t('profile.editProfile')}</Text>
              <TouchableOpacity onPress={() => setEditModalVisible(false)}>
                <Ionicons name="close" size={24} color={COLORS.textPrimary} />
              </TouchableOpacity>
            </View>

            <ScrollView
              contentContainerStyle={styles.modalBody}
              keyboardShouldPersistTaps="handled"
              showsVerticalScrollIndicator={false}
            >
              <AppInput
                label={t('auth.fullName')}
                value={editName}
                onChangeText={setEditName}
                autoCapitalize="words"
              />

              <Text style={styles.fieldLabel}>{t('auth.preferredLanguage')}</Text>
              <LanguageSelector value={editLanguage} onChange={setEditLanguage} />

              <Text style={[styles.fieldLabel, { marginTop: SPACING.md }]}>
                {t('auth.investorType')}
              </Text>
              <View style={styles.pillRow}>
                {investorOptions.map((o) => (
                  <OptionPill
                    key={o.value}
                    label={o.label}
                    selected={editInvestorType === o.value}
                    onPress={() => setEditInvestorType(o.value)}
                  />
                ))}
              </View>

              <Text style={[styles.fieldLabel, { marginTop: SPACING.md }]}>
                {t('auth.riskAppetite')}
              </Text>
              <View style={styles.pillRow}>
                {riskOptions.map((o) => (
                  <OptionPill
                    key={o.value}
                    label={o.label}
                    selected={editRisk === o.value}
                    onPress={() => setEditRisk(o.value)}
                  />
                ))}
              </View>

              <AppButton
                title={saving ? t('common.loading') : t('profile.saveChanges')}
                onPress={handleSaveProfile}
                loading={saving}
                style={{ marginTop: SPACING.xl }}
              />
            </ScrollView>
          </SafeAreaView>
        </View>
      </Modal>
    </SafeAreaView>
  );
};

// ─── Styles ───────────────────────────────────────────────

const pillStyles = StyleSheet.create({
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
});

const styles = StyleSheet.create({
  safeArea: {
    flex: 1,
    backgroundColor: COLORS.primary,
    paddingTop: Platform.OS === 'android' ? StatusBar.currentHeight : 0,
  },
  setupBanner: {
    backgroundColor: '#FEF3C7',
    paddingHorizontal: SPACING.lg,
    paddingVertical: SPACING.md,
    flexDirection: 'row',
    alignItems: 'flex-start',
    gap: SPACING.sm,
    marginBottom: SPACING.sm,
  },
  setupBannerText: {
    flex: 1,
  },
  setupBannerTitle: {
    fontSize: FONTS.md,
    fontWeight: '700',
    color: '#92400E',
    marginBottom: 4,
  },
  setupBannerDesc: {
    fontSize: FONTS.sm,
    color: '#78350F',
    lineHeight: 18,
  },
  headerBanner: {
    backgroundColor: COLORS.primary,
    alignItems: 'center',
    paddingTop: SPACING.xl,
    paddingBottom: SPACING.xxl,
    paddingHorizontal: SPACING.xl,
  },
  avatarCircle: {
    width: 88,
    height: 88,
    borderRadius: 44,
    backgroundColor: COLORS.accent,
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: SPACING.md,
    borderWidth: 3,
    borderColor: 'rgba(255,255,255,0.3)',
  },
  avatarText: {
    fontSize: FONTS.xxxl,
    fontWeight: '800',
    color: COLORS.white,
  },
  profileName: {
    fontSize: FONTS.xl,
    fontWeight: '800',
    color: COLORS.white,
    marginBottom: 4,
  },
  profileEmail: {
    fontSize: FONTS.sm,
    color: 'rgba(255,255,255,0.7)',
    marginBottom: SPACING.md,
  },
  badgeRow: {
    flexDirection: 'row',
    gap: SPACING.sm,
  },
  badge: {
    backgroundColor: 'rgba(30,144,255,0.3)',
    paddingHorizontal: SPACING.md,
    paddingVertical: 4,
    borderRadius: RADIUS.full,
    borderWidth: 1,
    borderColor: 'rgba(30,144,255,0.5)',
  },
  riskBadge: {
    backgroundColor: 'rgba(34,197,94,0.2)',
    borderColor: 'rgba(34,197,94,0.4)',
  },
  badgeText: {
    fontSize: FONTS.xs,
    fontWeight: '700',
    color: COLORS.white,
  },
  section: {
    padding: SPACING.xl,
    backgroundColor: COLORS.surface,
  },
  sectionTitle: {
    fontSize: FONTS.lg,
    fontWeight: '700',
    color: COLORS.textPrimary,
    marginBottom: SPACING.md,
  },
  infoCard: {
    backgroundColor: COLORS.card,
    borderRadius: RADIUS.lg,
    ...SHADOWS.small,
    overflow: 'hidden',
  },
  infoRow: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: SPACING.lg,
    gap: SPACING.md,
  },
  infoIconWrapper: {
    width: 36,
    height: 36,
    borderRadius: 18,
    backgroundColor: COLORS.accentLight,
    alignItems: 'center',
    justifyContent: 'center',
  },
  infoContent: {
    flex: 1,
  },
  infoLabel: {
    fontSize: FONTS.xs,
    color: COLORS.textMuted,
    fontWeight: '500',
    textTransform: 'uppercase',
    letterSpacing: 0.5,
  },
  infoValue: {
    fontSize: FONTS.md,
    color: COLORS.textPrimary,
    fontWeight: '600',
    marginTop: 2,
  },
  divider: {
    height: 1,
    backgroundColor: COLORS.divider,
    marginHorizontal: SPACING.lg,
  },
  actionButton: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: COLORS.card,
    borderRadius: RADIUS.lg,
    padding: SPACING.lg,
    marginBottom: SPACING.md,
    gap: SPACING.md,
    ...SHADOWS.small,
  },
  actionText: {
    flex: 1,
    fontSize: FONTS.md,
    fontWeight: '600',
    color: COLORS.textPrimary,
  },
  logoutButton: {
    marginTop: SPACING.sm,
  },
  // Modal styles
  modalOverlay: {
    flex: 1,
    backgroundColor: 'rgba(0,0,0,0.5)',
    justifyContent: 'flex-end',
  },
  modalSheet: {
    backgroundColor: COLORS.background,
    borderTopLeftRadius: RADIUS.xl,
    borderTopRightRadius: RADIUS.xl,
    maxHeight: '90%',
    ...SHADOWS.large,
  },
  modalHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: SPACING.xl,
    paddingVertical: SPACING.lg,
    borderBottomWidth: 1,
    borderBottomColor: COLORS.divider,
  },
  modalTitle: {
    fontSize: FONTS.lg,
    fontWeight: '700',
    color: COLORS.textPrimary,
  },
  modalBody: {
    padding: SPACING.xl,
    paddingBottom: SPACING.xxxl,
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
});

export default ProfileScreen;
