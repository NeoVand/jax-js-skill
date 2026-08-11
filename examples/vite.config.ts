import { defineConfig } from 'vite';
import { resolve } from 'node:path';

// The examples import the skill's templates directly, so Vite must be allowed
// to serve files from the repo root.
export default defineConfig({
	// jax-js lazily imports its wasm/webgpu backends, which is code-splitting.
	// Vite's default worker format is 'iife', which cannot code-split — the dev
	// server works and `vite build` fails.
	worker: { format: 'es' },
	optimizeDeps: { include: ['@jax-js/jax', '@jax-js/optax'] },
	server: { fs: { allow: [resolve(__dirname, '..')] } },
	build: {
		rollupOptions: {
			input: {
				main: resolve(__dirname, 'index.html'),
				sine: resolve(__dirname, 'sine.html'),
				gpt: resolve(__dirname, 'gpt.html'),
				bench: resolve(__dirname, 'bench.html')
			}
		}
	}
});
