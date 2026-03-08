-- ============================================
-- Supabase: Fix Existing Users with NULL Values
-- Run this AFTER creating the profiles table
-- This will update existing profiles that have NULL investor_type or risk_appetite
-- ============================================

-- 1. Check current profiles (for debugging)
SELECT 
  id, 
  name, 
  email, 
  language, 
  investor_type, 
  risk_appetite, 
  created_at 
FROM public.profiles;

-- 2. Update NULL values to defaults for existing users
UPDATE public.profiles
SET 
  investor_type = COALESCE(investor_type, 'Retail'),
  risk_appetite = COALESCE(risk_appetite, 'Medium'),
  language = COALESCE(language, 'en')
WHERE 
  investor_type IS NULL 
  OR risk_appetite IS NULL 
  OR language IS NULL;

-- 3. Verify the fix
SELECT 
  id, 
  name, 
  email, 
  language, 
  investor_type, 
  risk_appetite, 
  created_at 
FROM public.profiles;

-- 4. (Optional) If you want to delete all test users and start fresh:
-- WARNING: This will DELETE all users! Uncomment only if you want to start clean.
-- DELETE FROM public.profiles;
-- (Note: You'll also need to delete users from auth.users in Supabase Auth dashboard)
