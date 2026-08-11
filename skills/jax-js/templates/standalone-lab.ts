// The whole jax-js recipe in one file, on the main thread: pytree params,
// valueAndGrad over the tree, Adam from optax, a jitted step, and a canvas that
// redraws as the fit improves.
//
// Use this shape for teaching demos and models under ~100k parameters. Anything
// larger belongs in a Worker (templates/worker.ts) — a main-thread readback
// blocks layout and paint, and the page will jank.
//
// The pacing rule that keeps it smooth: do the arithmetic in bursts, then
// `await new Promise((r) => setTimeout(r))` so the browser can paint.

import { init, defaultDevice, jit, valueAndGrad, numpy as np, tree } from '@jax-js/jax';
import { adam, applyUpdates } from '@jax-js/optax';

/* eslint-disable @typescript-eslint/no-explicit-any */

export interface LabHandle {
	stop(): void;
	done: Promise<void>;
}

export interface LabOptions {
	canvas: HTMLCanvasElement;
	log?: (line: string) => void;
	steps?: number;
	layers?: number[];
	lr?: number;
	seed?: number;
	target?: (x: number) => number;
}

function mulberry32(seed: number) {
	let a = seed >>> 0;
	return () => {
		a |= 0;
		a = (a + 0x6d2b79f5) | 0;
		let t = Math.imul(a ^ (a >>> 15), 1 | a);
		t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
		return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
	};
}

/** Params as a plain object of arrays — that is all a pytree is. */
function initParams(layers: number[], seed: number) {
	const rand = mulberry32(seed);
	const w: any[] = [];
	const b: any[] = [];
	for (let i = 0; i < layers.length - 1; i++) {
		const [fin, fout] = [layers[i], layers[i + 1]];
		const limit = Math.sqrt(6 / (fin + fout)); // Glorot
		const buf = new Float32Array(fin * fout);
		for (let j = 0; j < buf.length; j++) buf[j] = (rand() * 2 - 1) * limit;
		w.push(np.array(buf).reshape([fin, fout]));
		b.push(np.zeros([fout]));
	}
	return { w, b };
}

/** Forward fold. The last layer stays linear — a tanh there would clamp the
 *  output to (−1, 1) and quietly cap what the network can express. */
function forward(p: any, x: any) {
	let h = x;
	for (let k = 0; k < p.w.length; k++) {
		h = np.dot(h, p.w[k]).add(p.b[k]);
		if (k < p.w.length - 1) h = np.tanh(h);
	}
	return h;
}

const lossFn = (p: any, x: any, y: any) => np.mean(np.square(forward(p, x).sub(y)));

/** Read a CSS custom property at draw time, so the canvas follows light/dark. */
const token = (el: HTMLElement, name: string, fallback: string) =>
	getComputedStyle(el).getPropertyValue(name).trim() || fallback;

export function runSineLab(opts: LabOptions): LabHandle {
	const {
		canvas,
		log = () => {},
		steps = 4000,
		layers = [1, 32, 32, 1],
		lr = 3e-3,
		seed = 7,
		target = (x: number) => Math.sin(3.1 * x)
	} = opts;

	let stopped = false;
	const done = (async () => {
		const devices = await init();
		const device = devices.includes('webgpu')
			? 'webgpu'
			: devices.includes('wasm')
				? 'wasm'
				: 'cpu';
		defaultDevice(device as any);
		log(`device: ${device} · network ${layers.join(' → ')}`);

		// ── data: points of a target curve, with a little noise ─────────────────
		const N = 256;
		const xs = new Float32Array(N);
		const ys = new Float32Array(N);
		const rand = mulberry32(seed + 1);
		for (let i = 0; i < N; i++) {
			xs[i] = (i / (N - 1)) * 2 - 1;
			ys[i] = target(xs[i]) + (rand() - 0.5) * 0.05;
		}
		const x = np.array(xs).reshape([N, 1]);
		const y = np.array(ys).reshape([N, 1]);

		let params: any = initParams(layers, seed);
		const solver = adam(lr, { b1: 0.9, b2: 0.99 });
		let optState = solver.init(tree.ref(params));

		// jit: the first call traces and compiles; every later call is a dispatch.
		// params go in as an ARGUMENT — a closure would bake them in as constants.
		const jitStep = jit((p: any, xx: any, yy: any) =>
			valueAndGrad((pp: any) => lossFn(pp, xx, yy))(p)
		);
		const jitPredict = jit((p: any, xx: any) => forward(p, xx));

		// ── drawing ──────────────────────────────────────────────────────────────
		const dpr = Math.min(2, window.devicePixelRatio || 1);
		const ctx = canvas.getContext('2d')!;
		function draw(pred: Float32Array) {
			const W = canvas.clientWidth;
			const H = 220;
			if (canvas.width !== W * dpr || canvas.height !== H * dpr) {
				canvas.width = W * dpr;
				canvas.height = H * dpr;
				canvas.style.height = `${H}px`;
			}
			ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
			ctx.clearRect(0, 0, W, H);
			const px = (v: number) => ((v + 1) / 2) * W;
			const py = (v: number) => H / 2 - v * (H / 2.6);

			ctx.setLineDash([4, 4]);
			ctx.strokeStyle = token(canvas, '--ink-3', '#a3a094');
			ctx.beginPath();
			for (let i = 0; i < N; i++) {
				const [X, Y] = [px(xs[i]), py(target(xs[i]))];
				i === 0 ? ctx.moveTo(X, Y) : ctx.lineTo(X, Y);
			}
			ctx.stroke();

			ctx.setLineDash([]);
			ctx.strokeStyle = token(canvas, '--accent', '#2b45d8');
			ctx.lineWidth = 2;
			ctx.beginPath();
			for (let i = 0; i < N; i++) {
				const [X, Y] = [px(xs[i]), py(pred[i])];
				i === 0 ? ctx.moveTo(X, Y) : ctx.lineTo(X, Y);
			}
			ctx.stroke();
			ctx.lineWidth = 1;
		}

		// ── the loop ─────────────────────────────────────────────────────────────
		for (let step = 1; step <= steps && !stopped; step++) {
			const [lossVal, grads] = jitStep(tree.ref(params), x.ref, y.ref);
			const [updates, nextState] = solver.update(grads, optState, tree.ref(params));
			params = applyUpdates(params, updates);
			optState = nextState;

			if (step % 100 === 0) {
				const pred = jitPredict(tree.ref(params), x.ref).dataSync() as Float32Array;
				draw(pred);
				log(`step ${String(step).padStart(4)}  loss ${lossVal.item().toFixed(5)}`);
				await new Promise((r) => setTimeout(r)); // let the page paint
			} else {
				lossVal.dispose(); // not read this step — free it explicitly
			}
		}

		// ── teardown ─────────────────────────────────────────────────────────────
		x.dispose();
		y.dispose();
		jitStep.dispose();
		jitPredict.dispose();
		tree.dispose(params);
		tree.dispose(optState);
		log(stopped ? 'stopped.' : 'done.');
	})();

	return {
		stop() {
			stopped = true;
		},
		done
	};
}
