import { defineConfig } from 'vite';
import { svelte } from '@sveltejs/vite-plugin-svelte';
import { resolve } from 'path';

export default defineConfig({
  plugins: [svelte()],
  root: '.',
  resolve: {
    alias: {
      '$lib': resolve('./src/lib'),
      '$components': resolve('./src/components'),
      '$views': resolve('./src/views'),
    },
  },
  build: {
    outDir: '../static/svelte',
    emptyOutDir: true,
    chunkSizeWarningLimit: 600,
    rollupOptions: {
      input: resolve('./src/main.js'),
      output: {
        entryFileNames: 'js/[name].js',
        chunkFileNames: 'js/[name]-[hash].js',
        assetFileNames: (info) => {
          if (info.name && info.name.endsWith('.css')) {
            return 'css/[name][extname]';
          }
          return 'assets/[name]-[hash][extname]';
        },
        manualChunks: {
          // Vendor libs in their own chunk for cache longevity
          'vendor-chart': ['chart.js'],
          'vendor-marked': ['marked'],
          'vendor-hljs': [
            'highlight.js/lib/core',
            'highlight.js/lib/languages/bash',
            'highlight.js/lib/languages/css',
            'highlight.js/lib/languages/javascript',
            'highlight.js/lib/languages/json',
            'highlight.js/lib/languages/markdown',
            'highlight.js/lib/languages/plaintext',
            'highlight.js/lib/languages/powershell',
            'highlight.js/lib/languages/python',
            'highlight.js/lib/languages/shell',
            'highlight.js/lib/languages/typescript',
            'highlight.js/lib/languages/xml',
            'highlight.js/lib/languages/yaml',
          ],
        },
      },
    },
  },
  server: {
    port: 5173,
    proxy: {
      '/api': 'http://localhost:5000',
      '/health': 'http://localhost:5000',
      '/static': 'http://localhost:5000',
    },
  },
});
