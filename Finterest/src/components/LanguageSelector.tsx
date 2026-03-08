/**
 * components/LanguageSelector.tsx
 * Dropdown component for selecting the app language.
 * Available as a modal picker for all 6 supported languages.
 */

import { Ionicons } from '@expo/vector-icons';
import React, { useState } from 'react';
import { useTranslation } from 'react-i18next';
import {
    FlatList,
    Modal,
    SafeAreaView,
    StyleSheet,
    Text,
    TouchableOpacity,
    View,
} from 'react-native';
import { COLORS, FONTS, RADIUS, SHADOWS, SPACING } from '../constants/theme';
import { saveLanguage } from '../i18n';

interface Language {
  code: string;
  label: string;
  nativeLabel: string;
}

const LANGUAGES: Language[] = [
  { code: 'en', label: 'English', nativeLabel: 'English' },
  { code: 'ta', label: 'Tamil', nativeLabel: 'தமிழ்' },
  { code: 'hi', label: 'Hindi', nativeLabel: 'हिन्दी' },
  { code: 'ml', label: 'Malayalam', nativeLabel: 'മലയാളം' },
  { code: 'te', label: 'Telugu', nativeLabel: 'తెలుగు' },
  { code: 'mr', label: 'Marathi', nativeLabel: 'मराठी' },
];

interface LanguageSelectorProps {
  /** Current language code */
  value: string;
  /** Called with new language code when selection changes */
  onChange?: (code: string) => void;
  /** If true, also persists & applies the i18n change immediately */
  applyGlobally?: boolean;
}

const LanguageSelector: React.FC<LanguageSelectorProps> = ({
  value,
  onChange,
  applyGlobally = false,
}) => {
  const { i18n, t } = useTranslation();
  const [modalVisible, setModalVisible] = useState(false);

  const currentLang = LANGUAGES.find((l) => l.code === value) || LANGUAGES[0];

  const handleSelect = async (lang: Language) => {
    setModalVisible(false);
    onChange?.(lang.code);

    if (applyGlobally) {
      // Change i18n language and persist to AsyncStorage
      await i18n.changeLanguage(lang.code);
      await saveLanguage(lang.code);
    }
  };

  return (
    <>
      {/* Trigger button */}
      <TouchableOpacity
        style={styles.selector}
        onPress={() => setModalVisible(true)}
        activeOpacity={0.8}
      >
        <Ionicons name="globe-outline" size={18} color={COLORS.accent} />
        <Text style={styles.selectorText}>
          {currentLang.nativeLabel}
        </Text>
        <Ionicons name="chevron-down" size={16} color={COLORS.textSecondary} />
      </TouchableOpacity>

      {/* Language picker modal */}
      <Modal
        visible={modalVisible}
        animationType="slide"
        transparent
        onRequestClose={() => setModalVisible(false)}
      >
        <View style={styles.overlay}>
          <SafeAreaView style={styles.sheet}>
            {/* Sheet header */}
            <View style={styles.sheetHeader}>
              <Text style={styles.sheetTitle}>{t('language.select')}</Text>
              <TouchableOpacity onPress={() => setModalVisible(false)}>
                <Ionicons name="close" size={24} color={COLORS.textPrimary} />
              </TouchableOpacity>
            </View>

            {/* Language list */}
            <FlatList
              data={LANGUAGES}
              keyExtractor={(item) => item.code}
              renderItem={({ item }) => {
                const isSelected = item.code === value;
                return (
                  <TouchableOpacity
                    style={[styles.languageItem, isSelected && styles.selectedItem]}
                    onPress={() => handleSelect(item)}
                    activeOpacity={0.7}
                  >
                    <View style={styles.languageInfo}>
                      <Text style={[styles.nativeLabel, isSelected && styles.selectedText]}>
                        {item.nativeLabel}
                      </Text>
                      <Text style={styles.langLabel}>{item.label}</Text>
                    </View>
                    {isSelected && (
                      <Ionicons name="checkmark-circle" size={22} color={COLORS.accent} />
                    )}
                  </TouchableOpacity>
                );
              }}
              ItemSeparatorComponent={() => <View style={styles.separator} />}
            />
          </SafeAreaView>
        </View>
      </Modal>
    </>
  );
};

const styles = StyleSheet.create({
  selector: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: COLORS.accentLight,
    paddingHorizontal: SPACING.md,
    paddingVertical: SPACING.sm,
    borderRadius: RADIUS.full,
    gap: SPACING.xs,
  },
  selectorText: {
    fontSize: FONTS.sm,
    fontWeight: '600',
    color: COLORS.primary,
  },
  overlay: {
    flex: 1,
    backgroundColor: 'rgba(0,0,0,0.5)',
    justifyContent: 'flex-end',
  },
  sheet: {
    backgroundColor: COLORS.background,
    borderTopLeftRadius: RADIUS.xl,
    borderTopRightRadius: RADIUS.xl,
    paddingBottom: SPACING.xxl,
    ...SHADOWS.large,
  },
  sheetHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: SPACING.xl,
    paddingVertical: SPACING.lg,
    borderBottomWidth: 1,
    borderBottomColor: COLORS.divider,
  },
  sheetTitle: {
    fontSize: FONTS.lg,
    fontWeight: '700',
    color: COLORS.textPrimary,
  },
  languageItem: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: SPACING.xl,
    paddingVertical: SPACING.md,
  },
  selectedItem: {
    backgroundColor: COLORS.accentLight,
  },
  languageInfo: {
    gap: 2,
  },
  nativeLabel: {
    fontSize: FONTS.md,
    fontWeight: '600',
    color: COLORS.textPrimary,
  },
  langLabel: {
    fontSize: FONTS.sm,
    color: COLORS.textSecondary,
  },
  selectedText: {
    color: COLORS.accent,
  },
  separator: {
    height: 1,
    backgroundColor: COLORS.divider,
    marginHorizontal: SPACING.xl,
  },
});

export default LanguageSelector;
