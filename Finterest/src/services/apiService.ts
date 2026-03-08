/**
 * services/apiService.ts
 * Centralized Axios API service for FinTraceQA.
 * Handles financial news and chatbot API calls.
 */

import axios, { AxiosInstance } from 'axios';

// Base API URL — points to the FinTraceQA FastAPI backend
// Use LAN IP when testing on a physical device (Expo Go / dev build).
// localhost only works on the same machine (web preview / emulator).
const BASE_URL = __DEV__
  ? 'http://10.70.41.101:8000'   // LAN IP — update if you change networks (run ipconfig)
  : 'https://your-production-url.com';  // swap when deploying

// Axios instance with shared config
const apiClient: AxiosInstance = axios.create({
  baseURL: BASE_URL,
  timeout: 30000,   // 30s — Gemini + Neo4j can take a few seconds
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor — attach auth tokens if needed
apiClient.interceptors.request.use(
  (config) => {
    // Example: config.headers.Authorization = `Bearer ${token}`;
    return config;
  },
  (error) => Promise.reject(error)
);

// Response interceptor — normalize errors
apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    const status = error?.response?.status;
    const message = error?.response?.data?.detail
      || error?.response?.data?.message
      || error.message
      || 'Unknown error';
    // Use console.warn (not console.error) so Expo dev overlay doesn't trigger
    console.warn(`[API Error] ${status ?? 'NETWORK'}: ${message}`);
    return Promise.reject(new Error(message));
  }
);

// ─── News API ─────────────────────────────────────────────

export interface NewsArticle {
  id: string;
  title: string;
  summary: string;
  source: string;
  timestamp: string;
  url?: string;
}

/**
 * Fetches financial news articles.
 * Returns mocked data if the API is unavailable.
 */
export const fetchFinancialNews = async (): Promise<NewsArticle[]> => {
  try {
    const response = await apiClient.get<NewsArticle[]>('/financial-news');
    return response.data;
  } catch {
    // Return mock data for development / when API is unavailable
    return getMockNews();
  }
};

// ─── Chatbot API ───────────────────────────────────────────

/** Shape the UI sends to the backend */
export interface ChatRequest {
  userId: string;
  query: string;
  language: string;  // "en" | "hi" | "mr" | "ta" | "te" | "kn" | "ml" | "auto"
}

/** Shape the UI consumes from the FinTraceQA backend */
export interface ChatResponse {
  answer: string;
  answer_lang: string;   // "en"|"hi"|"mr"|"ta"|"te"|"kn"|"ml" — actual language of answer
  explanation: string;   // joined explanation_steps
  confidence: number;
  warnings?: string[];
}

/** Full backend response (superset of ChatResponse) */
interface BackendChatResponse {
  answer: string;
  answer_lang: string;
  entities: { id: string; label: string; name: string; score: number }[];
  reasoning_paths: {
    path_id: string;
    hops: number;
    triples: { subj: string; rel: string; obj: string; time?: string; source?: string }[];
    score: number;
  }[];
  explanation_steps: string[];
  citations: { triple_index: number; source_uri?: string }[];
  confidence: number;
  warnings: string[];
}

/**
 * Sends a chat message to the FinTraceQA FastAPI backend.
 * Maps userId → session_id, query → message, language → lang.
 * For any non-English language ("hi"|"mr"|"ta"|"te"|"kn"|"ml"), the backend:
 *   1. Translates the native-script query to English for KG search (via DeepSeek)
 *   2. Translates the English answer back to the user's language before returning
 * Falls back to a mock response if the API is unavailable.
 */
export const sendChatMessage = async (payload: ChatRequest): Promise<ChatResponse> => {
  try {
    const response = await apiClient.post<BackendChatResponse>('/chat', {
      session_id: payload.userId,
      message: payload.query,
      lang: payload.language,   // pass through: "en"|"hi"|"mr"|"ta"|"te"|"kn"|"ml"|"auto"
    });
    const data = response.data;
    return {
      answer: data.answer,
      answer_lang: data.answer_lang,
      explanation: data.explanation_steps.join('\n'),
      confidence: data.confidence,
      warnings: data.warnings,
    };
  } catch {
    return getMockChatResponse(payload.query);
  }
};

// ─── Supported language codes ──────────────────────────────

/** ISO 639-1 codes supported by the translation pipeline */
export type SupportedLangCode = 'en' | 'hi' | 'mr' | 'ta' | 'te' | 'kn' | 'ml';

export const LANG_NAMES: Record<SupportedLangCode, string> = {
  en: 'English',
  hi: 'Hindi',
  mr: 'Marathi',
  ta: 'Tamil',
  te: 'Telugu',
  kn: 'Kannada',
  ml: 'Malayalam',
};

// ─── Translation API ───────────────────────────────────────

export interface TranslateRequest {
  text: string;
  source_lang: SupportedLangCode;
  target_lang: SupportedLangCode;
}

export interface TranslateResponse {
  translated: string;
  source_lang: string;
  target_lang: string;
}

/**
 * Translate any text between English and any supported Indian language via DeepSeek.
 * Supported: en, hi, mr, ta, te, kn, ml.
 * Useful for translating news summaries, UI labels, etc.
 */
export const translateText = async (payload: TranslateRequest): Promise<string> => {
  try {
    const response = await apiClient.post<TranslateResponse>('/translate', payload);
    return response.data.translated;
  } catch {
    // On failure return original text — never crash the UI
    return payload.text;
  }
};

// ─── Mock Data ─────────────────────────────────────────────

const getMockNews = (): NewsArticle[] => [
  {
    id: '1',
    title: 'Sensex Hits Record High Amid Global Rally',
    summary:
      'The BSE Sensex crossed the 80,000 mark for the first time as global equities rallied on positive macroeconomic data from the US.',
    source: 'Economic Times',
    timestamp: new Date(Date.now() - 1800000).toISOString(),
    url: 'https://example.com/news/1',
  },
  {
    id: '2',
    title: 'RBI Keeps Repo Rate Unchanged at 6.5%',
    summary:
      'The Reserve Bank of India maintained its benchmark lending rate, citing persistent inflation concerns while signaling a possible cut in Q3.',
    source: 'Mint',
    timestamp: new Date(Date.now() - 3600000).toISOString(),
    url: 'https://example.com/news/2',
  },
  {
    id: '3',
    title: 'IT Sector Leads Market Gains on Strong Q4 Earnings',
    summary:
      'Top IT companies including TCS and Infosys reported better-than-expected quarterly results, driving a sector-wide rally.',
    source: 'Business Standard',
    timestamp: new Date(Date.now() - 7200000).toISOString(),
    url: 'https://example.com/news/3',
  },
  {
    id: '4',
    title: 'Gold Prices Rise as Dollar Weakens',
    summary:
      'Gold futures climbed to a three-month high as the US dollar index fell following softer-than-expected inflation data.',
    source: 'Financial Express',
    timestamp: new Date(Date.now() - 10800000).toISOString(),
    url: 'https://example.com/news/4',
  },
  {
    id: '5',
    title: 'Crude Oil Edges Lower on Supply Concerns',
    summary:
      "Brent crude fell 1.2% amid concerns about weakening demand from China, while OPEC's decision on output cuts remains awaited.",
    source: 'Reuters',
    timestamp: new Date(Date.now() - 14400000).toISOString(),
    url: 'https://example.com/news/5',
  },
  {
    id: '6',
    title: 'Mutual Fund SIP Inflows Hit All-Time High',
    summary:
      'Monthly SIP contributions to Indian mutual funds crossed ₹20,000 crore for the first time, reflecting strong retail investor participation.',
    source: 'Moneycontrol',
    timestamp: new Date(Date.now() - 18000000).toISOString(),
    url: 'https://example.com/news/6',
  },
];

const getMockChatResponse = (query: string): ChatResponse => ({
  answer: `Based on the available financial data, your query about "${query}" relates to key market fundamentals. Current indicators suggest a cautiously optimistic outlook with moderate volatility expected in the near term.`,
  answer_lang: 'en',
  explanation:
    'This answer was derived by analyzing recent market trends, macroeconomic indicators, and historical patterns.',
  confidence: 0.78,
});

export default apiClient;
