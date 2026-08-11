#!/usr/bin/env node
// Write a runnable jax-js starter: vite + TypeScript + a Web Worker that owns
// the model, an engine on the main thread, and a page that trains and paints.
//
//   node scripts/scaffold.mjs my-app            # worker + MLP (default)
//   node scripts/scaffold.mjs my-app --standalone   # single file, main thread
//
// Then:  cd my-app && npm install && npm run dev

import { mkdirSync, writeFileSync, existsSync, copyFileSync } from 'node:fs';
import { join, dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const templates = resolve(here, '..', 'templates');

const args = process.argv.slice(2);
const standalone = args.includes('--standalone');
const target = args.find((a) => !a.startsWith('--'));

if (!target) {
	console.error('usage: node scripts/scaffold.mjs <dir> [--standalone]');
	process.exit(1);
}
const root = resolve(process.cwd(), target);
if (existsSync(root)) {
	console.error(`✗ ${root} already exists`);
	process.exit(1);
}

const write = (rel, content) => {
	const p = join(root, rel);
	mkdirSync(dirname(p), { recursive: true });
	writeFileSync(p, content);
	console.log(`  ${rel}`);
};
const copyTemplate = (name, rel = `src/${name}`) => {
	mkdirSync(dirname(join(root, rel)), { recursive: true });
	copyFileSync(join(templates, name), join(root, rel));
	console.log(`  ${rel}`);
};

console.log(`Creating ${target}/`);
mkdirSync(root, { recursive: true });

write(
	'package.json',
	JSON.stringify(
		{
			name: target.replace(/[^a-z0-9-]/gi, '-').toLowerCase(),
			private: true,
			version: '0.1.0',
			type: 'module',
			scripts: { dev: 'vite', build: 'vite build', preview: 'vite preview' },
			dependencies: { '@jax-js/jax': '^0.1.21', '@jax-js/optax': '^0.1.2' },
			devDependencies: { typescript: '^5.9.0', vite: '^7.0.0' }
		},
		null,
		2
	) + '\n'
);

write(
	'vite.config.ts',
	`import { defineConfig } from 'vite';

export default defineConfig({
	// jax-js lazily imports its wasm/webgpu backends, which is code-splitting.
	// Vite's default worker format is 'iife', which cannot code-split — the dev
	// server works and \`vite build\` fails with
	//   Invalid value "iife" for option "worker.format"
	worker: { format: 'es' },
	// Pre-bundle so the first page load is not hundreds of module requests.
	optimizeDeps: { include: ['@jax-js/jax', '@jax-js/optax'] }
});
`
);

write(
	'tsconfig.json',
	JSON.stringify(
		{
			compilerOptions: {
				target: 'ES2022',
				module: 'ESNext',
				moduleResolution: 'bundler',
				lib: ['ES2022', 'DOM', 'DOM.Iterable', 'WebWorker'],
				strict: true,
				skipLibCheck: true,
				noEmit: true,
				types: ['vite/client']
			},
			include: ['src']
		},
		null,
		2
	) + '\n'
);

write(
	'index.html',
	`<!doctype html>
<html lang="en">
	<head>
		<meta charset="utf-8" />
		<meta name="viewport" content="width=device-width, initial-scale=1" />
		<title>${target}</title>
		<style>
			:root { color-scheme: light dark; --paper:#faf9f5; --ink:#1d1c18; --ink-3:#a3a094;
				--line:#e5e2d8; --accent:#2b45d8; --warm:#d3541f; }
			@media (prefers-color-scheme: dark) { :root { --paper:#141310; --ink:#e9e6dc;
				--ink-3:#6c695e; --line:#2e2c24; --accent:#93a3ff; --warm:#ff8e57; } }
			body { margin:0 auto; max-width:70ch; padding:2.5rem 1.25rem;
				background:var(--paper); color:var(--ink);
				font:13px/1.6 ui-monospace,'SF Mono',Menlo,monospace; }
			canvas { width:100%; border:1px solid var(--line); border-radius:6px; }
			button { font:inherit; font-size:11px; padding:.25rem .7rem; border-radius:5px;
				border:1px solid var(--line); background:transparent; color:var(--ink); cursor:pointer; }
			pre { white-space:pre-wrap; color:var(--ink-3); }
		</style>
	</head>
	<body>
		<h1>${target}</h1>
		<p><button id="go">Train</button> <span id="status"></span></p>
		<canvas id="stage"></canvas>
		<pre id="out">booting…</pre>
		<script type="module" src="/src/main.ts"></script>
	</body>
</html>
`
);

if (standalone) {
	copyTemplate('standalone-lab.ts');
	write(
		'src/main.ts',
		`import { runSineLab, type LabHandle } from './standalone-lab';

const canvas = document.getElementById('stage') as HTMLCanvasElement;
const out = document.getElementById('out') as HTMLPreElement;
const go = document.getElementById('go') as HTMLButtonElement;

const lines: string[] = [];
const log = (s: string) => { lines.push(s); out.textContent = lines.slice(-12).join('\\n'); };

let lab: LabHandle | null = null;
go.addEventListener('click', () => {
	if (lab) { lab.stop(); lab = null; go.textContent = 'Train'; return; }
	lines.length = 0;
	go.textContent = 'Stop';
	lab = runSineLab({ canvas, log, steps: 4000 });
	void lab.done.then(() => { lab = null; go.textContent = 'Train'; });
});
`
	);
} else {
	copyTemplate('model-mlp.ts');
	copyTemplate('fused-adam.ts');
	write(
		'src/worker.ts',
		`// Owns jax-js, the params and the data. Adapt handlers; keep the dispatcher.
import { init, defaultDevice, numpy as np, nn, jit, valueAndGrad, tree } from '@jax-js/jax';
import { adam, applyUpdates } from '@jax-js/optax';
import * as mlp from './model-mlp';

/* eslint-disable @typescript-eslint/no-explicit-any */
interface RpcRequest { id: number; op: string; [k: string]: unknown }
const post = (msg: unknown, transfer?: Transferable[]) =>
	(self as unknown as Worker).postMessage(msg, { transfer: transfer ?? [] });

let cfg: any = null, params: any = null, optState: any = null, solver: any = null;
let x: any = null, y: any = null, jitStep: any = null, jitPredict: any = null;
let stopRequested = false, step = 0, device = 'none';

const handlers: Record<string, (r: RpcRequest) => unknown | Promise<unknown>> = {
	async init(req) {
		const devices = await init();
		// wasm is often FASTER than webgpu for small models — measure yours.
		device = devices.includes('webgpu') ? 'webgpu' : devices.includes('wasm') ? 'wasm' : 'cpu';
		defaultDevice(device as any);
		cfg = req.config;
		params = mlp.initParams(cfg);
		solver = adam((req.lr as number) ?? 3e-3);
		optState = solver.init(tree.ref(params));
		const n = req.n as number;
		x = np.array(new Float32Array(req.x as ArrayBuffer)).reshape([n, cfg.layers[0]]);
		y = np.array(new Float32Array(req.y as ArrayBuffer)).reshape([n, cfg.layers.at(-1)]);
		jitStep = jit((p: any, xx: any, yy: any) =>
			valueAndGrad((pp: any) => mlp.lossFn(pp, cfg, xx, yy))(p));
		jitPredict = jit((p: any, xx: any) => mlp.forward(p, cfg, xx).out);
		step = 0;
		return { device, paramCount: mlp.paramCount(params) };
	},
	async train(req) {
		stopRequested = false;
		const steps = (req.steps as number) ?? 200;
		for (let i = 0; i < steps; i++) {
			if (stopRequested) break;
			const t0 = performance.now();
			const [l, g] = jitStep(tree.ref(params), x.ref, y.ref);
			const [u, st] = solver.update(g, optState, tree.ref(params));
			params = applyUpdates(params, u);
			optState = st;
			step++;
			post({ id: req.id, event: 'metrics',
				m: { step, loss: l.item(), stepMs: performance.now() - t0 } });
			if (i % 4 === 3) await new Promise((r) => setTimeout(r, 0));  // let 'stop' land
		}
		return { step };
	},
	stop() { stopRequested = true; return {}; },
	predict() {
		const out = jitPredict(tree.ref(params), x.ref).dataSync() as Float32Array;
		const buf = out.slice();
		return { y: buf.buffer, __transfer: [buf.buffer] };
	},
	dispose() {
		tree.dispose(params); tree.dispose(optState);
		x?.dispose(); y?.dispose(); jitStep?.dispose(); jitPredict?.dispose();
		params = optState = x = y = null;
		return {};
	}
};

self.onmessage = async (e: MessageEvent<RpcRequest>) => {
	const req = e.data;
	try {
		const h = handlers[req.op];
		if (!h) throw new Error(\`unknown op: \${req.op}\`);
		const result = (await h(req)) as Record<string, unknown> & { __transfer?: Transferable[] };
		const transfer = result?.__transfer;
		if (transfer) delete result.__transfer;
		post({ id: req.id, ok: true, result }, transfer);
	} catch (err) {
		post({ id: req.id, ok: false,
			error: err instanceof Error ? \`\${err.name}: \${err.message}\` : String(err) });
	}
};
`
	);

	write(
		'src/engine.ts',
		`// Main-thread promise RPC. A plain class — never put this in reactive state.
export interface Metrics { step: number; loss: number; stepMs: number }
interface Pending { resolve: (v: unknown) => void; reject: (e: Error) => void;
	onMetrics?: (m: Metrics) => void }

export class Engine {
	private worker: Worker;
	private pending = new Map<number, Pending>();
	private nextId = 1;
	device = 'unknown';
	paramCount = 0;

	constructor() {
		this.worker = new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' });
		this.worker.onmessage = (e) => {
			const msg = e.data;
			const p = this.pending.get(msg.id);
			if (!p) return;
			if (msg.event === 'metrics') { p.onMetrics?.(msg.m); return; }
			this.pending.delete(msg.id);
			msg.ok ? p.resolve(msg.result) : p.reject(new Error(msg.error));
		};
		this.worker.onerror = (e) => {
			const err = new Error(e.message || 'worker error');
			for (const p of this.pending.values()) p.reject(err);
			this.pending.clear();
		};
	}

	private call<T>(op: string, payload: Record<string, unknown> = {},
		transfer: Transferable[] = [], onMetrics?: (m: Metrics) => void): Promise<T> {
		const id = this.nextId++;
		return new Promise<T>((resolve, reject) => {
			this.pending.set(id, { resolve: resolve as (v: unknown) => void, reject, onMetrics });
			this.worker.postMessage({ id, op, ...payload }, transfer);
		});
	}

	async init(config: unknown, x: Float32Array, y: Float32Array, n: number, lr?: number) {
		const xc = x.slice(), yc = y.slice();   // transferred buffers detach — copy
		const r = await this.call<{ device: string; paramCount: number }>(
			'init', { config, x: xc.buffer, y: yc.buffer, n, lr }, [xc.buffer, yc.buffer]);
		this.device = r.device;
		this.paramCount = r.paramCount;
	}
	train(steps: number, onMetrics: (m: Metrics) => void) {
		return this.call('train', { steps }, [], onMetrics).then(() => undefined);
	}
	stop() { return this.call('stop'); }
	async predict(): Promise<Float32Array> {
		const r = await this.call<{ y: ArrayBuffer }>('predict');
		return new Float32Array(r.y);
	}
	async dispose() {
		try { await Promise.race([this.call('dispose'), new Promise((r) => setTimeout(r, 400))]); }
		finally { this.worker.terminate(); this.pending.clear(); }
	}
}
`
	);

	write(
		'src/main.ts',
		`import { Engine } from './engine';

const canvas = document.getElementById('stage') as HTMLCanvasElement;
const out = document.getElementById('out') as HTMLPreElement;
const go = document.getElementById('go') as HTMLButtonElement;
const status = document.getElementById('status') as HTMLSpanElement;

// data: a sine wave to fit
const N = 256;
const xs = new Float32Array(N), ys = new Float32Array(N);
for (let i = 0; i < N; i++) {
	xs[i] = (i / (N - 1)) * 2 - 1;
	ys[i] = Math.sin(3.1 * xs[i]);
}

const cfg = { layers: [1, 32, 32, 1], activation: 'tanh', loss: 'mse', seed: 7 };

// A plain field, not framework state — it holds a Worker.
let engine: Engine | null = null;
let playing = false;

function draw(pred: Float32Array) {
	const dpr = Math.min(2, window.devicePixelRatio || 1);
	const W = canvas.clientWidth, H = 220;
	if (canvas.width !== W * dpr) { canvas.width = W * dpr; canvas.height = H * dpr;
		canvas.style.height = H + 'px'; }
	const ctx = canvas.getContext('2d')!;
	ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
	ctx.clearRect(0, 0, W, H);
	const token = (n: string, f: string) =>
		getComputedStyle(canvas).getPropertyValue(n).trim() || f;
	const px = (v: number) => ((v + 1) / 2) * W;
	const py = (v: number) => H / 2 - v * (H / 2.6);
	ctx.setLineDash([4, 4]); ctx.strokeStyle = token('--ink-3', '#a3a094');
	ctx.beginPath();
	for (let i = 0; i < N; i++) { const [X, Y] = [px(xs[i]), py(ys[i])];
		i ? ctx.lineTo(X, Y) : ctx.moveTo(X, Y); }
	ctx.stroke();
	ctx.setLineDash([]); ctx.strokeStyle = token('--accent', '#2b45d8'); ctx.lineWidth = 2;
	ctx.beginPath();
	for (let i = 0; i < N; i++) { const [X, Y] = [px(xs[i]), py(pred[i])];
		i ? ctx.lineTo(X, Y) : ctx.moveTo(X, Y); }
	ctx.stroke(); ctx.lineWidth = 1;
}

async function boot() {
	engine = new Engine();
	await engine.init(cfg, xs, ys, N, 5e-3);
	out.textContent = \`device: \${engine.device} · \${engine.paramCount} params\`;
	draw(await engine.predict());
}

go.addEventListener('click', async () => {
	if (!engine) return;
	if (playing) { playing = false; await engine.stop(); go.textContent = 'Train'; return; }
	playing = true; go.textContent = 'Pause';
	while (playing && engine) {
		await engine.train(50, (m) => {
			status.textContent = \`step \${m.step} · loss \${m.loss.toFixed(5)} · \${m.stepMs.toFixed(1)} ms\`;
		});
		if (!playing || !engine) break;
		draw(await engine.predict());
	}
});

window.addEventListener('pagehide', () => { playing = false; void engine?.dispose(); });
void boot();
`
	);
}

write(
	'README.md',
	`# ${target}

Built from the [jax-js skill](https://github.com/NeoVand/jax-js-skill) scaffold.

\`\`\`sh
npm install
npm run dev
\`\`\`

${standalone ? 'Everything is in `src/standalone-lab.ts` — main thread, small models.' : 'The model lives in `src/worker.ts`; `src/engine.ts` is the main-thread RPC client.'}

Before changing anything, read the five laws in the skill's SKILL.md —
especially that jax-js arrays are **moved, not shared**, and that \`.item()\`
consumes the array it reads.
`
);

write('.gitignore', 'node_modules\ndist\n');

console.log('');
console.log(`Done. Next:\n  cd ${target}\n  npm install\n  npm run dev`);
