import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// The client never hard-codes a host: /api is proxied to this project's
// FastAPI port, so the same code works in dev and behind a reverse proxy.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5107,
    proxy: {
      '/api': { target: 'http://127.0.0.1:8007', changeOrigin: true },
    },
  },
});
