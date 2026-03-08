/**
 * supabase/authService.ts
 * Supabase Authentication service functions.
 */

import { supabase } from './config';

export interface UserProfile {
  id: string;
  name: string;
  email: string;
  language: string;
  investor_type: 'Retail' | 'Institutional' | 'Student';
  risk_appetite: 'Low' | 'Medium' | 'High';
  created_at?: string;
}

/**
 * Creates a new user with email/password and saves profile to profiles table.
 */
export const signUpUser = async (
  email: string,
  password: string,
  profile: Omit<UserProfile, 'id' | 'created_at'>
): Promise<any> => {
  console.log('[signUpUser] Starting signup with profile:', profile);
  
  // 1. Sign up the user
  const { data: authData, error: authError } = await supabase.auth.signUp({
    email,
    password,
  });

  if (authError) {
    console.error('[signUpUser] Auth error:', authError);
    throw authError;
  }
  if (!authData.user) {
    console.error('[signUpUser] No user returned from signup');
    throw new Error('Signup failed: no user returned');
  }

  console.log('[signUpUser] User created successfully, ID:', authData.user.id);

  // 2. Create profile record with explicit values (avoid undefined)
  const profileData = {
    id: authData.user.id,
    name: profile.name,
    email: email,
    language: profile.language || 'en',
    investor_type: profile.investor_type || 'Retail',
    risk_appetite: profile.risk_appetite || 'Medium',
    created_at: new Date().toISOString(),
  };

  console.log('[signUpUser] Upserting profile:', profileData);

  const { data: insertedProfile, error: profileError } = await supabase
    .from('profiles')
    .upsert(profileData, { onConflict: 'id' })  // UPSERT: update if exists, insert if not
    .select()
    .single();

  if (profileError) {
    console.error('[signUpUser] Profile insert error:', profileError);
    throw new Error(`Profile creation failed: ${profileError.message}`);
  }

  console.log('[signUpUser] Profile created successfully:', insertedProfile);

  return authData.user;
};

/**
 * Signs in an existing user with email/password.
 */
export const loginUser = async (
  email: string,
  password: string
): Promise<any> => {
  const { data, error } = await supabase.auth.signInWithPassword({
    email,
    password,
  });

  if (error) throw error;
  return data.user;
};

/**
 * Signs out the currently authenticated user.
 */
export const logoutUser = async (): Promise<void> => {
  const { error } = await supabase.auth.signOut();
  if (error) throw error;
};

/**
 * Fetches user profile from the profiles table.
 */
export const getUserProfile = async (
  uid: string
): Promise<UserProfile | null> => {
  console.log('[getUserProfile] Fetching profile for user:', uid);
  
  const { data, error } = await supabase
    .from('profiles')
    .select('*')
    .eq('id', uid)
    .maybeSingle(); // Use maybeSingle() instead of single() to handle "0 rows" gracefully

  if (error) {
    console.warn('[getUserProfile] Error:', error);
    return null;
  }

  if (!data) {
    console.warn('[getUserProfile] No profile found for user:', uid);
    return null;
  }

  console.log('[getUserProfile] Profile found:', data);
  return data as UserProfile;
};

/**
 * Updates user profile in the profiles table.
 * Uses UPDATE only (profile always exists by signup time).
 */
export const updateUserProfile = async (
  uid: string,
  updates: Partial<UserProfile>
): Promise<void> => {
  console.log('[updateUserProfile] Updating profile for:', uid, updates);

  // Try UPDATE first
  const { data, error } = await supabase
    .from('profiles')
    .update(updates)
    .eq('id', uid)
    .select()
    .single();

  if (error) {
    console.error('[updateUserProfile] Error:', error);
    throw error;
  }

  console.log('[updateUserProfile] Updated successfully:', data);
};

/**
 * Subscribes to auth state changes.
 * Returns unsubscribe function.
 */
export const subscribeToAuthChanges = (
  callback: (user: any | null) => void
): (() => void) => {
  const { data: { subscription } } = supabase.auth.onAuthStateChange(
    (_event, session) => {
      callback(session?.user ?? null);
    }
  );

  return () => {
    subscription.unsubscribe();
  };
};
