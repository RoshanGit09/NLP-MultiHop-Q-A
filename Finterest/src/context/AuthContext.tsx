/**
 * context/AuthContext.tsx
 * React Context for global authentication state management using Supabase.
 * Wraps the entire app to provide user state to all components.
 */

import React, {
    createContext,
    ReactNode,
    useContext,
    useEffect,
    useState,
} from 'react';
import { getUserProfile, subscribeToAuthChanges, UserProfile } from '../firebase/authService';

// Shape of the auth context value
interface AuthContextType {
  user: any | null; // Supabase user object
  userProfile: UserProfile | null;
  isLoading: boolean;
  setUserProfile: (profile: UserProfile | null) => void;
}

// Create context with default values
const AuthContext = createContext<AuthContextType>({
  user: null,
  userProfile: null,
  isLoading: true,
  setUserProfile: () => {},
});

/**
 * AuthProvider wraps the app and manages Supabase auth state.
 * Fetches profile from profiles table whenever the auth user changes.
 */
export const AuthProvider = ({ children }: { children: ReactNode }) => {
  const [user, setUser] = useState<any | null>(null);
  const [userProfile, setUserProfile] = useState<UserProfile | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    // Subscribe to Supabase auth state changes
    // No delay needed — Supabase client works immediately!
    const unsubscribe = subscribeToAuthChanges(async (supabaseUser) => {
      setUser(supabaseUser);

      if (supabaseUser) {
        // Fetch the user's profile from profiles table
        try {
          const profile = await getUserProfile(supabaseUser.id);
          setUserProfile(profile);
        } catch (e) {
          console.warn('[AuthContext] Failed to fetch profile:', e);
          setUserProfile(null);
        }
      } else {
        // User logged out — clear profile
        setUserProfile(null);
      }

      setIsLoading(false);
    });

    return () => {
      unsubscribe();
    };
  }, []);

  return (
    <AuthContext.Provider value={{ user, userProfile, isLoading, setUserProfile }}>
      {children}
    </AuthContext.Provider>
  );
};

/**
 * Custom hook to consume the AuthContext.
 * Must be used inside <AuthProvider>.
 */
export const useAuth = (): AuthContextType => {
  return useContext(AuthContext);
};
