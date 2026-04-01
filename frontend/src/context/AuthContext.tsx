import { createContext, useContext, useReducer, useEffect, useCallback, type ReactNode } from 'react';

// API Base URL
const API_BASE = import.meta.env.VITE_API_BASE_URL || '/api';

// Types
interface User {
  id: number;
  username: string;
  created_at: string;
}

interface AuthState {
  user: User | null;
  token: string | null;
  isLoading: boolean;
  error: string | null;
  authDisabled: boolean;
}

type AuthAction =
  | { type: 'SET_LOADING'; payload: boolean }
  | { type: 'SET_USER'; payload: { user: User; token: string } }
  | { type: 'SET_ERROR'; payload: string }
  | { type: 'LOGOUT' }
  | { type: 'CLEAR_ERROR' }
  | { type: 'AUTH_DISABLED'; payload: User };

interface AuthContextValue {
  state: AuthState;
  login: (username: string, password: string) => Promise<boolean>;
  logout: () => void;
  clearError: () => void;
  isAuthenticated: boolean;
}

const TOKEN_KEY = 'auth_token';

// Initial state
const initialState: AuthState = {
  user: null,
  token: localStorage.getItem(TOKEN_KEY),
  isLoading: true,
  error: null,
  authDisabled: false,
};

// Reducer
function authReducer(state: AuthState, action: AuthAction): AuthState {
  switch (action.type) {
    case 'SET_LOADING':
      return { ...state, isLoading: action.payload };
    case 'SET_USER':
      return {
        ...state,
        user: action.payload.user,
        token: action.payload.token,
        isLoading: false,
        error: null,
      };
    case 'SET_ERROR':
      return { ...state, error: action.payload, isLoading: false };
    case 'LOGOUT':
      return { ...state, user: null, token: null, isLoading: false, error: null };
    case 'CLEAR_ERROR':
      return { ...state, error: null };
    case 'AUTH_DISABLED':
      return { ...state, user: action.payload, token: null, isLoading: false, error: null, authDisabled: true };
    default:
      return state;
  }
}

// Context
const AuthContext = createContext<AuthContextValue | null>(null);

// Provider
export function AuthProvider({ children }: { children: ReactNode }) {
  const [state, dispatch] = useReducer(authReducer, initialState);

  // Check auth status and verify token on mount
  useEffect(() => {
    const initAuth = async () => {
      try {
        // First check if auth is enabled on the backend
        const statusResponse = await fetch(`${API_BASE}/auth/status`);
        if (statusResponse.ok) {
          const { auth_enabled } = await statusResponse.json();

          if (!auth_enabled) {
            // Auth disabled — get default user info and skip login
            const meResponse = await fetch(`${API_BASE}/auth/me`);
            if (meResponse.ok) {
              const user = await meResponse.json();
              dispatch({ type: 'AUTH_DISABLED', payload: user });
            } else {
              // Can't get user info but auth is off — create a placeholder
              dispatch({ type: 'AUTH_DISABLED', payload: { id: 1, username: 'admin', created_at: '' } });
            }
            return;
          }
        }
      } catch {
        // Backend not reachable — fall through to normal auth flow
      }

      // Auth enabled (or status check failed) — verify token
      const token = localStorage.getItem(TOKEN_KEY);
      if (!token) {
        dispatch({ type: 'SET_LOADING', payload: false });
        return;
      }

      try {
        const response = await fetch(`${API_BASE}/auth/me`, {
          headers: {
            Authorization: `Bearer ${token}`,
          },
        });

        if (response.ok) {
          const user = await response.json();
          dispatch({ type: 'SET_USER', payload: { user, token } });
        } else {
          localStorage.removeItem(TOKEN_KEY);
          dispatch({ type: 'LOGOUT' });
        }
      } catch {
        localStorage.removeItem(TOKEN_KEY);
        dispatch({ type: 'LOGOUT' });
      }
    };

    initAuth();
  }, []);

  const login = useCallback(async (username: string, password: string): Promise<boolean> => {
    dispatch({ type: 'SET_LOADING', payload: true });
    dispatch({ type: 'CLEAR_ERROR' });

    try {
      const response = await fetch(`${API_BASE}/auth/login`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ username, password }),
      });

      if (!response.ok) {
        const data = await response.json().catch(() => ({}));
        dispatch({ type: 'SET_ERROR', payload: data.detail || 'Login failed' });
        return false;
      }

      const { access_token } = await response.json();
      localStorage.setItem(TOKEN_KEY, access_token);

      // Get user info
      const userResponse = await fetch(`${API_BASE}/auth/me`, {
        headers: {
          Authorization: `Bearer ${access_token}`,
        },
      });

      if (userResponse.ok) {
        const user = await userResponse.json();
        dispatch({ type: 'SET_USER', payload: { user, token: access_token } });
        return true;
      } else {
        dispatch({ type: 'SET_ERROR', payload: 'Failed to get user info' });
        return false;
      }
    } catch (err) {
      dispatch({ type: 'SET_ERROR', payload: 'Network error' });
      return false;
    }
  }, []);

  const logout = useCallback(() => {
    localStorage.removeItem(TOKEN_KEY);
    dispatch({ type: 'LOGOUT' });
  }, []);

  const clearError = useCallback(() => {
    dispatch({ type: 'CLEAR_ERROR' });
  }, []);

  const value: AuthContextValue = {
    state,
    login,
    logout,
    clearError,
    isAuthenticated: state.authDisabled || (!!state.user && !!state.token),
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

// Hook
export function useAuth() {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
}

// Export token getter for API calls
export function getAuthToken(): string | null {
  return localStorage.getItem(TOKEN_KEY);
}
