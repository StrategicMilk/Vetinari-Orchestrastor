import { vitePreprocess } from '@sveltejs/vite-plugin-svelte';

export default {
  preprocess: vitePreprocess(),
  compilerOptions: {
    // Svelte 5 runes mode is default; no legacy flag needed
    runes: true,
  },
};
