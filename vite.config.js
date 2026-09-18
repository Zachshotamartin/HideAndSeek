import { defineConfig } from 'vite';

// The shared figure is linked during development; use the host's Three instance.
export default defineConfig({ resolve: { dedupe: ['three'] } });
