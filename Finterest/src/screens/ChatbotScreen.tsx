/**
 * screens/ChatbotScreen.tsx
 * WhatsApp-style financial Q&A chatbot.
 * Features: user/bot bubbles, typing indicator, collapsible explanation, auto-scroll.
 */

import { Ionicons } from '@expo/vector-icons';
import React, { useCallback, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import {
    ActivityIndicator,
    FlatList,
    KeyboardAvoidingView,
    Platform,
    SafeAreaView,
    StatusBar,
    StyleSheet,
    Text,
    TextInput,
    TouchableOpacity,
    View,
} from 'react-native';
import { COLORS, FONTS, RADIUS, SHADOWS, SPACING } from '../constants/theme';
import { useAuth } from '../context/AuthContext';
import { sendChatMessage } from '../services/apiService';

// Shape of a single chat message
interface Message {
  id: string;
  role: 'user' | 'bot';
  text: string;
  explanation?: string;
  confidence?: number;
  timestamp: Date;
  showExplanation?: boolean;
}

/** Formats a Date object as HH:MM */
const formatMessageTime = (date: Date): string => {
  return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
};

// ─── Message Bubble Components ────────────────────────────

const UserBubble: React.FC<{ message: Message }> = ({ message }) => (
  <View style={bubbleStyles.userContainer}>
    <View style={bubbleStyles.userBubble}>
      <Text style={bubbleStyles.userText}>{message.text}</Text>
      <Text style={bubbleStyles.userTime}>{formatMessageTime(message.timestamp)}</Text>
    </View>
  </View>
);

const BotBubble: React.FC<{
  message: Message;
  onToggleExplanation: (id: string) => void;
  t: (key: string) => string;
}> = ({ message, onToggleExplanation, t }) => (
  <View style={bubbleStyles.botContainer}>
    {/* Bot avatar */}
    <View style={bubbleStyles.botAvatar}>
      <Text style={bubbleStyles.botAvatarText}>AI</Text>
    </View>

    <View style={bubbleStyles.botBubble}>
      {/* Main answer */}
      <Text style={bubbleStyles.botText}>{message.text}</Text>

      {/* Confidence score */}
      {message.confidence !== undefined && (
        <View style={bubbleStyles.confidenceRow}>
          <Ionicons name="analytics-outline" size={12} color={COLORS.accent} />
          <Text style={bubbleStyles.confidenceText}>
            {t('chatbot.confidence')}: {Math.round(message.confidence * 100)}%
          </Text>
        </View>
      )}

      {/* Collapsible explanation */}
      {message.explanation ? (
        <View style={bubbleStyles.explanationContainer}>
          <TouchableOpacity
            style={bubbleStyles.explanationToggle}
            onPress={() => onToggleExplanation(message.id)}
            activeOpacity={0.8}
          >
            <Ionicons
              name={message.showExplanation ? 'chevron-up' : 'chevron-down'}
              size={14}
              color={COLORS.accent}
            />
            <Text style={bubbleStyles.explanationToggleText}>
              {message.showExplanation ? t('chatbot.hideReasoning') : t('chatbot.viewReasoning')}
            </Text>
          </TouchableOpacity>

          {/* Collapsible content */}
          {message.showExplanation && (
            <View style={bubbleStyles.explanationContent}>
              <Text style={bubbleStyles.explanationText}>{message.explanation}</Text>
            </View>
          )}
        </View>
      ) : null}

      <Text style={bubbleStyles.botTime}>{formatMessageTime(message.timestamp)}</Text>
    </View>
  </View>
);

// ─── Typing Indicator ─────────────────────────────────────

const TypingIndicator: React.FC<{ label: string }> = ({ label }) => (
  <View style={bubbleStyles.botContainer}>
    <View style={bubbleStyles.botAvatar}>
      <Text style={bubbleStyles.botAvatarText}>AI</Text>
    </View>
    <View style={[bubbleStyles.botBubble, styles.typingBubble]}>
      <ActivityIndicator size="small" color={COLORS.accent} />
      <Text style={styles.typingText}>{label}</Text>
    </View>
  </View>
);

// ─── Main Screen ──────────────────────────────────────────

const ChatbotScreen: React.FC = () => {
  const { t, i18n } = useTranslation();
  const { user } = useAuth();
  const flatListRef = useRef<FlatList>(null);

  const [messages, setMessages] = useState<Message[]>([
    {
      id: 'greeting',
      role: 'bot',
      text: t('chatbot.greeting'),
      timestamp: new Date(),
    },
  ]);
  const [inputText, setInputText] = useState('');
  const [isTyping, setIsTyping] = useState(false);

  // Toggle explanation expansion for a bot message
  const handleToggleExplanation = useCallback((id: string) => {
    setMessages((prev) =>
      prev.map((m) =>
        m.id === id ? { ...m, showExplanation: !m.showExplanation } : m
      )
    );
  }, []);

  // Send message and call chatbot API
  const handleSend = async () => {
    const query = inputText.trim();
    if (!query) return;

    const userMsg: Message = {
      id: `user_${Date.now()}`,
      role: 'user',
      text: query,
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMsg]);
    setInputText('');
    setIsTyping(true);

    // Scroll to bottom after user message
    setTimeout(() => flatListRef.current?.scrollToEnd({ animated: true }), 100);

    try {
      const response = await sendChatMessage({
        userId: user?.uid || 'anonymous',
        query,
        language: i18n.language,
      });

      const botMsg: Message = {
        id: `bot_${Date.now()}`,
        role: 'bot',
        text: response.answer,
        explanation: response.explanation,
        confidence: response.confidence,
        timestamp: new Date(),
        showExplanation: false,
      };

      setMessages((prev) => [...prev, botMsg]);
    } catch {
      const errorMsg: Message = {
        id: `err_${Date.now()}`,
        role: 'bot',
        text: t('chatbot.error'),
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, errorMsg]);
    } finally {
      setIsTyping(false);
      setTimeout(() => flatListRef.current?.scrollToEnd({ animated: true }), 100);
    }
  };

  return (
    <SafeAreaView style={styles.safeArea}>
      <StatusBar barStyle="light-content" backgroundColor={COLORS.primary} />

      {/* Header */}
      <View style={styles.header}>
        <View style={styles.headerLeft}>
          <View style={styles.headerAvatar}>
            <Ionicons name="analytics" size={20} color={COLORS.white} />
          </View>
          <View>
            <Text style={styles.headerTitle}>{t('chatbot.title')}</Text>
            <Text style={styles.headerStatus}>● Online</Text>
          </View>
        </View>
      </View>

      <KeyboardAvoidingView
        style={styles.flex}
        behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
        keyboardVerticalOffset={Platform.OS === 'ios' ? 0 : 20}
      >
        {/* Message list */}
        <FlatList
          ref={flatListRef}
          data={messages}
          keyExtractor={(item) => item.id}
          contentContainerStyle={styles.messageList}
          showsVerticalScrollIndicator={false}
          onContentSizeChange={() => flatListRef.current?.scrollToEnd({ animated: false })}
          renderItem={({ item }) =>
            item.role === 'user' ? (
              <UserBubble message={item} />
            ) : (
              <BotBubble
                message={item}
                onToggleExplanation={handleToggleExplanation}
                t={t}
              />
            )
          }
          ListFooterComponent={
            isTyping ? <TypingIndicator label={t('chatbot.typing')} /> : null
          }
        />

        {/* Input area */}
        <View style={styles.inputBar}>
          <TextInput
            style={styles.textInput}
            placeholder={t('chatbot.placeholder')}
            placeholderTextColor={COLORS.textMuted}
            value={inputText}
            onChangeText={setInputText}
            multiline
            maxLength={500}
            returnKeyType="send"
            onSubmitEditing={handleSend}
          />
          <TouchableOpacity
            style={[styles.sendButton, !inputText.trim() && styles.sendButtonDisabled]}
            onPress={handleSend}
            disabled={!inputText.trim() || isTyping}
            activeOpacity={0.8}
          >
            <Ionicons name="send" size={20} color={COLORS.white} />
          </TouchableOpacity>
        </View>
      </KeyboardAvoidingView>
    </SafeAreaView>
  );
};

// ─── Shared bubble styles ─────────────────────────────────

const bubbleStyles = StyleSheet.create({
  userContainer: {
    alignItems: 'flex-end',
    marginVertical: SPACING.xs,
    paddingHorizontal: SPACING.lg,
  },
  userBubble: {
    backgroundColor: COLORS.userBubble,
    borderRadius: RADIUS.lg,
    borderBottomRightRadius: 4,
    padding: SPACING.md,
    maxWidth: '78%',
  },
  userText: {
    fontSize: FONTS.md,
    color: COLORS.userBubbleText,
    lineHeight: 20,
  },
  userTime: {
    fontSize: 10,
    color: 'rgba(255,255,255,0.7)',
    textAlign: 'right',
    marginTop: SPACING.xs,
  },
  botContainer: {
    flexDirection: 'row',
    alignItems: 'flex-end',
    marginVertical: SPACING.xs,
    paddingHorizontal: SPACING.lg,
    gap: SPACING.sm,
  },
  botAvatar: {
    width: 32,
    height: 32,
    borderRadius: 16,
    backgroundColor: COLORS.primary,
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 2,
  },
  botAvatarText: {
    fontSize: 10,
    fontWeight: '800',
    color: COLORS.white,
  },
  botBubble: {
    backgroundColor: COLORS.botBubble,
    borderRadius: RADIUS.lg,
    borderBottomLeftRadius: 4,
    padding: SPACING.md,
    maxWidth: '78%',
  },
  botText: {
    fontSize: FONTS.md,
    color: COLORS.botBubbleText,
    lineHeight: 20,
  },
  botTime: {
    fontSize: 10,
    color: COLORS.textMuted,
    textAlign: 'right',
    marginTop: SPACING.xs,
  },
  confidenceRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
    marginTop: SPACING.xs,
  },
  confidenceText: {
    fontSize: 11,
    color: COLORS.accent,
    fontWeight: '600',
  },
  explanationContainer: {
    marginTop: SPACING.sm,
    borderTopWidth: 1,
    borderTopColor: COLORS.border,
    paddingTop: SPACING.sm,
  },
  explanationToggle: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
  },
  explanationToggleText: {
    fontSize: FONTS.xs,
    color: COLORS.accent,
    fontWeight: '600',
  },
  explanationContent: {
    marginTop: SPACING.sm,
    backgroundColor: 'rgba(30,144,255,0.06)',
    borderRadius: RADIUS.sm,
    padding: SPACING.sm,
  },
  explanationText: {
    fontSize: FONTS.sm,
    color: COLORS.textSecondary,
    lineHeight: 18,
    fontStyle: 'italic',
  },
});

// ─── Screen-level styles ──────────────────────────────────

const styles = StyleSheet.create({
  safeArea: {
    flex: 1,
    backgroundColor: COLORS.primary,
    paddingTop: Platform.OS === 'android' ? StatusBar.currentHeight : 0,
  },
  flex: { flex: 1, backgroundColor: COLORS.surface },
  header: {
    backgroundColor: COLORS.primary,
    paddingHorizontal: SPACING.lg,
    paddingVertical: SPACING.md,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
  },
  headerLeft: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: SPACING.sm,
  },
  headerAvatar: {
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: COLORS.accent,
    alignItems: 'center',
    justifyContent: 'center',
  },
  headerTitle: {
    fontSize: FONTS.md,
    fontWeight: '700',
    color: COLORS.white,
  },
  headerStatus: {
    fontSize: 11,
    color: COLORS.success,
    fontWeight: '500',
  },
  messageList: {
    paddingVertical: SPACING.md,
    paddingBottom: SPACING.lg,
  },
  typingBubble: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: SPACING.sm,
    paddingVertical: SPACING.sm,
  },
  typingText: {
    fontSize: FONTS.sm,
    color: COLORS.textSecondary,
    fontStyle: 'italic',
  },
  inputBar: {
    flexDirection: 'row',
    alignItems: 'flex-end',
    paddingHorizontal: SPACING.md,
    paddingVertical: SPACING.sm,
    backgroundColor: COLORS.background,
    borderTopWidth: 1,
    borderTopColor: COLORS.border,
    gap: SPACING.sm,
    ...SHADOWS.small,
  },
  textInput: {
    flex: 1,
    minHeight: 44,
    maxHeight: 120,
    backgroundColor: COLORS.surface,
    borderRadius: RADIUS.full,
    paddingHorizontal: SPACING.lg,
    paddingVertical: SPACING.sm,
    fontSize: FONTS.md,
    color: COLORS.textPrimary,
    borderWidth: 1,
    borderColor: COLORS.border,
  },
  sendButton: {
    width: 44,
    height: 44,
    borderRadius: 22,
    backgroundColor: COLORS.accent,
    alignItems: 'center',
    justifyContent: 'center',
    ...SHADOWS.small,
  },
  sendButtonDisabled: {
    backgroundColor: COLORS.border,
  },
});

export default ChatbotScreen;
