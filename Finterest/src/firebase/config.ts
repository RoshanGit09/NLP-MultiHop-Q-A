/**
 * supabase/config.ts
 * Supabase client configuration.
 *
 * SETUP:
 * 1. Go to https://supabase.com and create a new project
 * 2. Get your project URL and anon key from Settings > API
 * 3. Create a .env file in the root directory (copy from .env.example)
 * 4. Add your credentials to .env:
 *    EXPO_PUBLIC_SUPABASE_URL=https://xxxxx.supabase.co
 *    EXPO_PUBLIC_SUPABASE_ANON_KEY=eyJhbGciOi...
 * 5. Restart the dev server: npx expo start --clear
 * 6. Enable Email auth in Authentication > Providers
 * 7. Create a 'profiles' table in Database (see README.md)
 */

import AsyncStorage from '@react-native-async-storage/async-storage';
import { createClient } from '@supabase/supabase-js';

// Expo automatically loads EXPO_PUBLIC_* variables from .env
const supabaseUrl = process.env.EXPO_PUBLIC_SUPABASE_URL || '';
const supabaseAnonKey = process.env.EXPO_PUBLIC_SUPABASE_ANON_KEY || '';

if (!supabaseUrl || !supabaseAnonKey) {
  console.error(
    '❌ Missing Supabase credentials!\n' +
    '1. Create a .env file in the root directory\n' +
    '2. Add:\n' +
    '   EXPO_PUBLIC_SUPABASE_URL=https://your-project.supabase.co\n' +
    '   EXPO_PUBLIC_SUPABASE_ANON_KEY=your-anon-key\n' +
    '3. Restart: npx expo start --clear'
  );
}

// Supabase client with AsyncStorage for session persistence
export const supabase = createClient(supabaseUrl, supabaseAnonKey, {
  auth: {
    storage: AsyncStorage,
    autoRefreshToken: true,
    persistSession: true,
    detectSessionInUrl: false,
  },
});

export default supabase;
