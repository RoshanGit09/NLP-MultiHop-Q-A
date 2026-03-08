/**
 * screens/NewsScreen.tsx
 * Financial news feed with pull-to-refresh, loading, and error states.
 * Fetches from API with mock fallback for development.
 */

import { Ionicons } from '@expo/vector-icons';
import { useFocusEffect } from '@react-navigation/native';
import React, { useCallback, useState } from 'react';
import { useTranslation } from 'react-i18next';
import {
    ActivityIndicator,
    FlatList,
    Linking,
    Platform,
    RefreshControl,
    SafeAreaView,
    StatusBar,
    StyleSheet,
    Text,
    TouchableOpacity,
    View,
} from 'react-native';
import AppCard from '../components/AppCard';
import { COLORS, FONTS, RADIUS, SPACING } from '../constants/theme';
import { fetchFinancialNews, NewsArticle } from '../services/apiService';

/** Formats an ISO timestamp into a human-readable relative time */
const formatTime = (isoString: string): string => {
  const now = Date.now();
  const diff = Math.floor((now - new Date(isoString).getTime()) / 1000);
  if (diff < 60) return 'Just now';
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
  if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
  return `${Math.floor(diff / 86400)}d ago`;
};

const NewsScreen: React.FC = () => {
  const { t } = useTranslation();
  const [articles, setArticles] = useState<NewsArticle[]>([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState(false);

  // Fetch news data
  const loadNews = async (isRefresh = false) => {
    if (isRefresh) setRefreshing(true);
    else setLoading(true);
    setError(false);

    try {
      const data = await fetchFinancialNews();
      setArticles(data);
    } catch {
      setError(true);
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  };

  // Load news when screen comes into focus
  useFocusEffect(
    useCallback(() => {
      loadNews();
    }, [])
  );

  // Handle "Read More" — open article URL in browser
  const handleReadMore = (url?: string) => {
    if (url) Linking.openURL(url).catch(() => {});
  };

  // Render a single news card
  const renderArticle = ({ item }: { item: NewsArticle }) => (
    <AppCard style={styles.card}>
      {/* Source + timestamp row */}
      <View style={styles.metaRow}>
        <View style={styles.sourceBadge}>
          <Text style={styles.sourceText}>{item.source}</Text>
        </View>
        <Text style={styles.timeText}>{formatTime(item.timestamp)}</Text>
      </View>

      {/* Title */}
      <Text style={styles.articleTitle}>{item.title}</Text>

      {/* Summary */}
      <Text style={styles.articleSummary} numberOfLines={3}>
        {item.summary}
      </Text>

      {/* Read More button */}
      <TouchableOpacity
        style={styles.readMoreButton}
        onPress={() => handleReadMore(item.url)}
        activeOpacity={0.8}
      >
        <Text style={styles.readMoreText}>{t('news.readMore')}</Text>
        <Ionicons name="arrow-forward" size={14} color={COLORS.accent} />
      </TouchableOpacity>
    </AppCard>
  );

  // Loading skeleton
  if (loading) {
    return (
      <View style={styles.centerContainer}>
        <ActivityIndicator size="large" color={COLORS.accent} />
        <Text style={styles.loadingText}>{t('news.loading')}</Text>
      </View>
    );
  }

  // Error state
  if (error) {
    return (
      <View style={styles.centerContainer}>
        <Ionicons name="cloud-offline-outline" size={56} color={COLORS.textMuted} />
        <Text style={styles.errorText}>{t('news.error')}</Text>
        <TouchableOpacity style={styles.retryButton} onPress={() => loadNews()}>
          <Text style={styles.retryText}>{t('news.retry')}</Text>
        </TouchableOpacity>
      </View>
    );
  }

  // Empty state
  if (articles.length === 0) {
    return (
      <View style={styles.centerContainer}>
        <Ionicons name="newspaper-outline" size={56} color={COLORS.textMuted} />
        <Text style={styles.emptyText}>{t('news.empty')}</Text>
      </View>
    );
  }

  return (
    <SafeAreaView style={styles.safeArea}>
      <StatusBar barStyle="dark-content" backgroundColor={COLORS.background} />
      <View style={styles.container}>
        <FlatList
          data={articles}
          keyExtractor={(item) => item.id}
          renderItem={renderArticle}
          contentContainerStyle={styles.listContent}
          showsVerticalScrollIndicator={false}
          refreshControl={
            <RefreshControl
              refreshing={refreshing}
              onRefresh={() => loadNews(true)}
              colors={[COLORS.accent]}
              tintColor={COLORS.accent}
            />
          }
          ListHeaderComponent={
            <View style={styles.listHeader}>
              <Text style={styles.screenTitle}>{t('news.title')}</Text>
              <Text style={styles.articleCount}>{articles.length} articles</Text>
            </View>
          }
        />
      </View>
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  safeArea: {
    flex: 1,
    backgroundColor: COLORS.surface,
    paddingTop: Platform.OS === 'android' ? StatusBar.currentHeight : 0,
  },
  container: {
    flex: 1,
    backgroundColor: COLORS.surface,
  },
  listContent: {
    paddingHorizontal: SPACING.lg,
    paddingBottom: SPACING.xxl,
  },
  listHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingVertical: SPACING.xl,
    paddingTop: SPACING.xxl,
  },
  screenTitle: {
    fontSize: FONTS.xl,
    fontWeight: '800',
    color: COLORS.textPrimary,
  },
  articleCount: {
    fontSize: FONTS.sm,
    color: COLORS.textSecondary,
    fontWeight: '500',
  },
  card: {
    marginBottom: SPACING.md,
  },
  metaRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: SPACING.sm,
  },
  sourceBadge: {
    backgroundColor: COLORS.accentLight,
    paddingHorizontal: SPACING.sm,
    paddingVertical: 3,
    borderRadius: RADIUS.sm,
  },
  sourceText: {
    fontSize: FONTS.xs,
    fontWeight: '700',
    color: COLORS.accent,
    textTransform: 'uppercase',
    letterSpacing: 0.5,
  },
  timeText: {
    fontSize: FONTS.xs,
    color: COLORS.textMuted,
  },
  articleTitle: {
    fontSize: FONTS.md,
    fontWeight: '700',
    color: COLORS.textPrimary,
    marginBottom: SPACING.sm,
    lineHeight: 22,
  },
  articleSummary: {
    fontSize: FONTS.sm,
    color: COLORS.textSecondary,
    lineHeight: 20,
    marginBottom: SPACING.md,
  },
  readMoreButton: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: SPACING.xs,
    alignSelf: 'flex-start',
  },
  readMoreText: {
    fontSize: FONTS.sm,
    color: COLORS.accent,
    fontWeight: '600',
  },
  centerContainer: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: COLORS.surface,
    gap: SPACING.md,
    padding: SPACING.xl,
  },
  loadingText: {
    fontSize: FONTS.md,
    color: COLORS.textSecondary,
    marginTop: SPACING.sm,
  },
  errorText: {
    fontSize: FONTS.md,
    color: COLORS.textSecondary,
    textAlign: 'center',
  },
  emptyText: {
    fontSize: FONTS.md,
    color: COLORS.textSecondary,
    textAlign: 'center',
  },
  retryButton: {
    backgroundColor: COLORS.accent,
    paddingHorizontal: SPACING.xl,
    paddingVertical: SPACING.sm,
    borderRadius: RADIUS.full,
  },
  retryText: {
    color: COLORS.white,
    fontWeight: '700',
    fontSize: FONTS.sm,
  },
});

export default NewsScreen;
