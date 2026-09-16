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
			dependencies: { '@jax-js/jax': '^0.1.24', '@jax-js/optax': '^0.1.2' },
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
		if (params) handlers.dispose({ id: req.id, op: 'dispose' });
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
		if (params) tree.dispose(params);
		if (optState) tree.dispose(optState);
		x?.dispose(); y?.dispose(); jitStep?.dispose(); jitPredict?.dispose();
		params = optState = x = y = jitStep = jitPredict = null;
		return {};
	}
};

async function dispatch(req: RpcRequest) {
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
}
let queue = Promise.resolve();
self.onmessage = (e: MessageEvent<RpcRequest>) => {
	const req = e.data;
	if (req.op === 'stop') {
		stopRequested = true;
		post({ id: req.id, ok: true, result: {} });
		return;
	}
	if (req.op === 'dispose') stopRequested = true;
	queue = queue.then(() => dispatch(req));
};
`
	);

	write(
		'src/engine.ts',
		`// Main-thread promise RPC. Keep this resource handle separate from UI metrics.
export interface Metrics { step: number; loss: number; stepMs: number }
interface Pending { resolve: (v: unknown) => void; reject: (e: Error) => void;
	onMetrics?: (m: Metrics) => void }

export class Engine {
	private worker: Worker;
	private pending = new Map<number, Pending>();
	private nextId = 1;
	private closed = false;
	private disposal: Promise<void> | null = null;
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
			this.shutdown(new Error(e.message || 'worker error'));
		};
		this.worker.onmessageerror = () => this.shutdown(new Error('worker message could not be decoded'));
	}

	private rejectPending(error: Error, exceptId?: number) {
		for (const [id, request] of this.pending) {
			if (id === exceptId) continue;
			this.pending.delete(id);
			request.reject(error);
		}
	}

	private shutdown(error: Error) {
		this.closed = true;
		this.rejectPending(error);
		this.worker.terminate();
	}

	private call<T>(
		op: string,
		payload: Record<string, unknown> = {},
		transfer: Transferable[] = [],
		onMetrics?: (m: Metrics) => void
	): Promise<T> {
		if (this.closed) return Promise.reject(new Error('Engine disposed or closed'));
		const id = this.nextId++;
		return new Promise<T>((resolve, reject) => {
			this.pending.set(id, { resolve: resolve as (v: unknown) => void, reject, onMetrics });
			try {
				this.worker.postMessage({ id, op, ...payload }, transfer);
			} catch (error) {
				this.pending.delete(id);
				reject(error);
			}
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
	/** Compilation can delay a graceful reply. Bound cleanup, reject callers,
	 *  and terminate even when the worker cannot process its disposal request. */
	dispose(): Promise<void> {
		if (this.disposal) return this.disposal;
		if (this.closed) return Promise.resolve();
		const disposeId = this.nextId;
		const reply = this.call('dispose');
		this.closed = true;
		this.rejectPending(new Error('Engine disposed'), disposeId);
		this.disposal = (async () => {
			let timer: ReturnType<typeof setTimeout> | undefined;
			try {
				await Promise.race([reply, new Promise<void>((resolve) => {
					timer = setTimeout(resolve, 400);
				})]);
			} catch {
				// Graceful cleanup is best-effort; termination below is unconditional.
			} finally {
				clearTimeout(timer);
				this.shutdown(new Error('Engine disposed'));
			}
		})();
		return this.disposal;
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

// Resource handle separate from the values rendered in the page.
let engine: Engine | null = null;
let playing = false;
let run = 0;
go.disabled = true;

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
	const e = new Engine();
	engine = e;
	try {
		await e.init(cfg, xs, ys, N, 5e-3);
		if (engine !== e) return;
		const predicted = await e.predict();
		if (engine !== e) return;
		out.textContent = \`device: \${e.device} · \${e.paramCount} params\`;
		draw(predicted);
		go.disabled = false;
	} catch (error) {
		if (engine !== e) return;
		engine = null;
		void e.dispose();
		out.textContent = String(error);
	}
}

go.addEventListener('click', async () => {
	const e = engine;
	if (!e) return;
	const myRun = ++run;
	if (playing) {
		playing = false;
		await e.stop().catch(() => {});
		if (engine === e && myRun === run) go.textContent = 'Train';
		return;
	}
	playing = true; go.textContent = 'Pause';
	try {
		while (playing && engine === e && myRun === run) {
			await e.train(50, (m) => {
				if (engine !== e || myRun !== run) return;
				status.textContent = \`step \${m.step} · loss \${m.loss.toFixed(5)} · \${m.stepMs.toFixed(1)} ms\`;
			});
			if (!playing || engine !== e || myRun !== run) break;
			const predicted = await e.predict();
			if (engine !== e || myRun !== run) return;
			draw(predicted);
		}
	} catch (error) {
		if (engine !== e || myRun !== run) return;
		playing = false; go.textContent = 'Train';
		out.textContent = String(error);
	}
});

window.addEventListener('pagehide', () => {
	playing = false; run++;
	const e = engine; engine = null;
	void e?.dispose();
});
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
