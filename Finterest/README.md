# FinTraceQA — Multilingual Financial Q&A Mobile App

A production-ready React Native (Expo) app for financial news and AI-powered Q&A with full multilingual support, powered by Supabase.

---

## 🌐 Supported Languages

| Code | Language   | Native Script |
|------|------------|---------------|
| `en` | English    | English       |
| `ta` | Tamil      | தமிழ்          |
| `hi` | Hindi      | हिन्दी         |
| `ml` | Malayalam  | മലയാളം        |
| `te` | Telugu     | తెలుగు         |
| `mr` | Marathi    | मराठी          |

---

## 📁 Project Structure

```
src/
├── App.tsx                    # Root component (i18n init + AuthProvider)
├── i18n.js                    # i18next configuration + AsyncStorage persistence
├── locales/
│   ├── en.json  ta.json  hi.json  ml.json  te.json  mr.json
├── firebase/                  # (renamed but kept for compatibility)
│   ├── config.ts              # ⚠️ Supabase client setup — add your URL & key here
│   └── authService.ts         # Supabase Auth + profiles table service functions
├── context/
│   └── AuthContext.tsx        # Global auth state (React Context)
├── navigation/
│   ├── AppNavigator.tsx       # Root navigator (Auth vs Main)
│   ├── AuthNavigator.tsx      # Stack: Login → Signup
│   └── MainNavigator.tsx      # Bottom Tabs: News | Chatbot | Profile
├── screens/
│   ├── LoginScreen.tsx
│   ├── SignupScreen.tsx
│   ├── NewsScreen.tsx
│   ├── ChatbotScreen.tsx
│   └── ProfileScreen.tsx
├── components/
│   ├── AppButton.tsx  AppInput.tsx  AppCard.tsx  LanguageSelector.tsx
├── services/
│   └── apiService.ts          # Axios client (news + chatbot)
└── constants/
    └── theme.ts               # Design tokens
```

---

## � Supabase Setup (Required before running)

### 1. Create Supabase Project
1. Go to [https://supabase.com](https://supabase.com) and sign in
2. Click **New Project** → Organization: Choose one → Name: `FinTraceQA`
3. Database Password: Create a strong password → Region: Choose nearest → **Create**
4. Wait ~2 minutes for project provisioning

### 2. Get API Credentials
1. Go to **Settings** (gear icon) → **API**
2. Copy **Project URL** and **`anon` `public`** key

### 3. Create `.env` File
1. Copy `.env.example` to `.env` in the root directory
2. Add your credentials:
```bash
EXPO_PUBLIC_SUPABASE_URL=https://xxxxx.supabase.co
EXPO_PUBLIC_SUPABASE_ANON_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```
3. **Restart the dev server**: `npx expo start --clear`

> ⚠️ Never commit `.env` to git! It's already in `.gitignore`.

### 4. Enable Email Authentication
1. Go to **Authentication** → **Providers**
2. Ensure **Email** is enabled (it's enabled by default)
3. Optional: Configure Email Templates under **Email Templates** if you want custom verification emails

### 5. Create Profiles Table
1. Go to **Table Editor** → **New Table**
2. Name: `profiles`
3. Add columns (click **Add column**):
   - `id` (uuid, primary key) — check **"Is Primary Key"** and **"Is Identity"**
   - `name` (text)
   - `email` (text)
   - `language` (text)
   - `investor_type` (text)
   - `risk_appetite` (text)
   - `created_at` (timestamptz, default: `now()`)
4. Click **Save**

### 6. Set Row Level Security (RLS)
1. In **Table Editor**, select the `profiles` table
2. Click **RLS** icon (shield) → **Enable RLS**
3. Click **New Policy** → "Enable read access for users based on user_id"
   - Policy name: `Users can view own profile`
   - Target roles: `authenticated`
   - USING expression:
     ```sql
     auth.uid() = id
     ```
   - WITH CHECK expression:
     ```sql
     auth.uid() = id
     ```
4. Click **Review** → **Save Policy**
5. Repeat for INSERT/UPDATE/DELETE policies or create a single policy with all operations enabled

### 7. Link Auth Users to Profiles (Optional Trigger)
This automatically creates a profile row when a user signs up:
1. Go to **Database** → **Functions** → **New Function**
2. Name: `handle_new_user`
3. Paste:
```sql
BEGIN
  INSERT INTO public.profiles (id, email, created_at)
  VALUES (new.id, new.email, now());
  RETURN new;
END;
```
4. Go to **Database** → **Triggers** → **New Trigger**
5. Name: `on_auth_user_created`
6. Table: `auth.users`
7. Events: `INSERT`
8. Type: `AFTER`
9. Function: `handle_new_user`
10. **Confirm**

---

## 📦 Installation

```bash
npm install
npm start          # Start Expo dev server
npm run android    # Run on Android device/emulator
npm run ios        # Run on iOS simulator (Mac only)
```

---

## 🔌 API Integration

Update `BASE_URL` in `src/services/apiService.ts` to point to your backend:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/financial-news` | GET | Returns news articles array |
| `/chat` | POST | `{ userId, query, language }` → `{ answer, explanation, confidence }` |

> Mock data is used as fallback when the API is unavailable.

---

## 🎨 Theme Colors

| Token | Value | Usage |
|-------|-------|-------|
| Primary | `#0A1F44` | Headers, navigation, primary buttons |
| Accent | `#1E90FF` | Interactive elements, links, badges |
| Background | `#FFFFFF` | Screen backgrounds |
| Error | `#EF4444` | Error states, logout |
| Success | `#22C55E` | Status indicators |

---

## ✅ Features

- ✅ **Supabase Email/Password Authentication** (no Firebase timing issues!)
- ✅ **Profiles table** with user data storage
- ✅ **AsyncStorage** session persistence across app restarts
- ✅ Auto-detect device language on first launch
- ✅ Persist + apply language changes via AsyncStorage
- ✅ Instant UI re-render on language switch (react-i18next)
- ✅ 6 fully translated languages including 5 Indian scripts
- ✅ Financial news feed (pull-to-refresh, loading, error states)
- ✅ WhatsApp-style chatbot with collapsible AI reasoning
- ✅ Protected routes based on auth state
- ✅ Edit profile with Supabase sync
- ✅ Modern financial dashboard UI
- ✅ **Works immediately** — no module timing errors!
