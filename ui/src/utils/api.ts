import axios from 'axios';
import { createGlobalState } from 'react-global-hooks';

export const isAuthorizedState = createGlobalState(false);

export const apiClient = axios.create();

// Add a request interceptor to add token from localStorage
apiClient.interceptors.request.use(config => {
  const token = localStorage.getItem('AI_TOOLKIT_AUTH');
  if (token) {
    config.headers['Authorization'] = `Bearer ${token}`;
  }
  return config;
});

export interface DatasetImage {
  img_path: string;
}

// The /api/datasets/listImages route returns a compact payload: a single shared `root`
// plus each file's sub-path (an upstream transfer optimization — see that route). Rebuild
// the full native path per entry so callers get the { img_path } objects the UI expects.
// Falls through unchanged if the payload is ever already in object form.
export function normalizeListImages(data: { root?: string; images?: unknown }): DatasetImage[] {
  const root = data?.root ?? '';
  const images = Array.isArray(data?.images) ? data.images : [];
  return images.map((entry: any) => (typeof entry === 'string' ? { img_path: root + entry } : entry));
}

// Add a response interceptor to handle 401 errors
apiClient.interceptors.response.use(
  response => response, // Return successful responses as-is
  error => {
    // Check if the error is a 401 Unauthorized
    if (error.response && error.response.status === 401) {
      // Clear the auth token from localStorage
      localStorage.removeItem('AI_TOOLKIT_AUTH');
      isAuthorizedState.set(false);
    }

    // Reject the promise with the error so calling code can still catch it
    return Promise.reject(error);
  },
);
