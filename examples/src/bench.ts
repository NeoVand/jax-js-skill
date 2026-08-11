// Measure the things the skill asserts, in a real browser.
//
//  1. device choice for a small MLP: cpu vs wasm vs webgpu
//  2. device choice for a transformer
//  3. optax-outside-jit vs a hand-fused Adam inside jit
//  4. sync-every-step vs let-it-run-ahead (upstream issue #151)
//
// Everything here is deliberately main-thread so the numbers are comparable.

import { init, defaultDevice, numpy as np, nn, jit, valueAndGrad, tree } from '@jax-js/jax';
import { adam, applyUpdates } from '@jax-js/optax';
import * as mlp from '../../skills/jax-js/templates/model-mlp';
import * as gpt from '../../skills/jax-js/templates/model-transformer';
import { fusedAdam } from '../../skills/jax-js/templates/fused-adam';

/* eslint-disable @typescript-eslint/no-explicit-any */

const out = document.getElementById('out') as HTMLPreElement;
const lines: string[] = [];
const log = (s = '') => {
	lines.push(s);
	out.textContent = lines.join('\n');
};
const yieldToPage = () => new Promise((r) => setTimeout(r, 0));

const median = (xs: number[]) => {
	const s = [...xs].sort((a, b) => a - b);
	return s[Math.floor(s.length / 2)];
};

// ── 1 & 2: device comparison ────────────────────────────────────────────────

async function benchMlp(device: string, steps = 60) {
	defaultDevice(device as any);
	const cfg = { layers: [2, 64, 64, 2], activation: 'tanh' as const, loss: 'mse' as const, seed: 7 };
	let params = mlp.initParams(cfg);
	const N = 256;
	const x = np.array(new Float32Array(N * 2).fill(0.3)).reshape([N, 2]);
	const y = np.array(new Float32Array(N * 2).fill(0.7)).reshape([N, 2]);
	const jitStep = jit((p: any, xx: any, yy: any) =>
		valueAndGrad((pp: any) => mlp.lossFn(pp, cfg, xx, yy))(p)
	);
	const solver = adam(1e-3);
	let st = solver.init(tree.ref(params));
	const ts: number[] = [];
	for (let i = 0; i < steps; i++) {
		const t0 = performance.now();
		const [l, g] = jitStep(tree.ref(params), x.ref, y.ref);
		const [u, st2] = solver.update(g, st, tree.ref(params));
		params = applyUpdates(params, u);
		st = st2;
		l.item();
		if (i > 4) ts.push(performance.now() - t0); // skip compile + warmup
		if (i % 10 === 9) await yieldToPage();
	}
	x.dispose();
	y.dispose();
	jitStep.dispose();
	tree.dispose(params);
	tree.dispose(st);
	return median(ts);
}

async function benchGpt(device: string, steps = 30) {
	defaultDevice(device as any);
	const cfg = { nLayer: 2, nEmbd: 96, nHead: 4, blockSize: 96, vocab: 24 };
	let params = gpt.initParams(cfg, 7);
	const data = new Uint16Array(20000);
	for (let i = 0; i < data.length; i++) data[i] = i % cfg.vocab;
	let s = 1;
	const rand = () => ((s = (s * 1664525 + 1013904223) >>> 0), s / 4294967296);
	const jitStep = jit((p: any, a: any, b: any, t: any) =>
		valueAndGrad((pp: any) => gpt.lossFn(pp, cfg, a, b, t))(p)
	);
	const solver = adam(1e-3);
	let st = solver.init(tree.ref(params));
	const ts: number[] = [];
	for (let i = 0; i < steps; i++) {
		const t0 = performance.now();
		const { tokenOH, posOH, targetOH } = gpt.makeBatchOH(cfg, data, rand, 8);
		const [l, g] = jitStep(tree.ref(params), tokenOH, posOH, targetOH);
		const [u, st2] = solver.update(g, st, tree.ref(params));
		params = applyUpdates(params, u);
		st = st2;
		l.item();
		if (i > 2) ts.push(performance.now() - t0);
		await yieldToPage();
	}
	jitStep.dispose();
	gpt.disposeTree(params);
	gpt.disposeTree(st);
	return { ms: median(ts), params: 2 * cfg.vocab * cfg.nEmbd + cfg.blockSize * cfg.nEmbd + cfg.nLayer * 12 * cfg.nEmbd * cfg.nEmbd };
}

// ── 3: optax outside jit vs fused adam inside jit ───────────────────────────

async function benchOptimizer(device: string, steps = 60) {
	defaultDevice(device as any);
	const cfg = { layers: [64, 128, 128, 64], activation: 'relu' as const, loss: 'mse' as const, seed: 3 };
	const N = 128;
	const mk = () => ({
		x: np.array(new Float32Array(N * 64).fill(0.2)).reshape([N, 64]),
		y: np.array(new Float32Array(N * 64).fill(0.4)).reshape([N, 64])
	});

	// A: optax outside jit
	let params: any = mlp.initParams(cfg);
	const jitStep = jit((p: any, xx: any, yy: any) =>
		valueAndGrad((pp: any) => mlp.lossFn(pp, cfg, xx, yy))(p)
	);
	const solver = adam(1e-3);
	let st = solver.init(tree.ref(params));
	const { x, y } = mk();
	const a: number[] = [];
	for (let i = 0; i < steps; i++) {
		const t0 = performance.now();
		const [l, g] = jitStep(tree.ref(params), x.ref, y.ref);
		const [u, st2] = solver.update(g, st, tree.ref(params));
		params = applyUpdates(params, u);
		st = st2;
		l.item();
		if (i > 4) a.push(performance.now() - t0);
		if (i % 10 === 9) await yieldToPage();
	}
	jitStep.dispose();
	tree.dispose(params);
	tree.dispose(st);

	// B: everything fused inside one jit
	let p2: any = mlp.initParams(cfg);
	const fused = fusedAdam((p: any, xx: any, yy: any) => mlp.lossFn(p, cfg, xx, yy), { lr: 1e-3 });
	let fs = fused.init(tree.ref(p2));
	const b: number[] = [];
	for (let i = 0; i < steps; i++) {
		const t0 = performance.now();
		const [l, np2, ns] = fused.step(p2, fs, x.ref, y.ref);
		p2 = np2;
		fs = ns;
		l.item();
		if (i > 4) b.push(performance.now() - t0);
		if (i % 10 === 9) await yieldToPage();
	}
	fused.dispose();
	tree.dispose(p2);
	tree.dispose(fs.m);
	tree.dispose(fs.v);
	x.dispose();
	y.dispose();
	return { optax: median(a), fused: median(b) };
}

// ── 5: the same comparison on a transformer ─────────────────────────────────

async function benchGptOptimizer(device: string, steps = 24) {
	defaultDevice(device as any);
	const cfg = { nLayer: 2, nEmbd: 96, nHead: 4, blockSize: 96, vocab: 24 };
	const data = new Uint16Array(20000);
	for (let i = 0; i < data.length; i++) data[i] = i % cfg.vocab;
	const mkRand = () => {
		let s = 1;
		return () => ((s = (s * 1664525 + 1013904223) >>> 0), s / 4294967296);
	};

	// A: optax outside jit
	let params: any = gpt.initParams(cfg, 7);
	const jitStep = jit((p: any, a: any, b: any, t: any) =>
		valueAndGrad((pp: any) => gpt.lossFn(pp, cfg, a, b, t))(p)
	);
	const solver = adam(1e-3);
	let st = solver.init(tree.ref(params));
	let rand = mkRand();
	const A: number[] = [];
	for (let i = 0; i < steps; i++) {
		const t0 = performance.now();
		const { tokenOH, posOH, targetOH } = gpt.makeBatchOH(cfg, data, rand, 8);
		const [l, g] = jitStep(tree.ref(params), tokenOH, posOH, targetOH);
		const [u, st2] = solver.update(g, st, tree.ref(params));
		params = applyUpdates(params, u);
		st = st2;
		l.item();
		if (i > 2) A.push(performance.now() - t0);
		await yieldToPage();
	}
	jitStep.dispose();
	gpt.disposeTree(params);
	gpt.disposeTree(st);

	// B: fused inside jit
	const B = await benchGptFusedRaw(cfg, data, steps);
	return { optax: median(A), fused: B };
}

async function benchGptFusedRaw(cfg: any, data: Uint16Array, steps: number) {
	let params: any = gpt.initParams(cfg, 7);
	const fused = fusedAdam(
		(p: any, a: any, b: any, t: any) => gpt.lossFn(p, cfg, a, b, t),
		{ lr: 1e-3 }
	);
	let fs = fused.init(tree.ref(params));
	let s = 1;
	const rand = () => ((s = (s * 1664525 + 1013904223) >>> 0), s / 4294967296);
	const ts: number[] = [];
	for (let i = 0; i < steps; i++) {
		const t0 = performance.now();
		const { tokenOH, posOH, targetOH } = gpt.makeBatchOH(cfg, data, rand, 8);
		const [l, p2, st2] = fused.step(params, fs, tokenOH, posOH, targetOH);
		params = p2;
		fs = st2;
		l.item();
		if (i > 2) ts.push(performance.now() - t0);
		await yieldToPage();
	}
	fused.dispose();
	gpt.disposeTree(params);
	gpt.disposeTree(fs.m);
	gpt.disposeTree(fs.v);
	return median(ts);
}

async function benchGptFused(device: string, cfg: any, steps = 16) {
	defaultDevice(device as any);
	const data = new Uint16Array(20000);
	for (let i = 0; i < data.length; i++) data[i] = i % cfg.vocab;
	return benchGptFusedRaw(cfg, data, steps);
}

// ── 4: sync every step vs run ahead ─────────────────────────────────────────

async function benchSync(device: string, steps = 40) {
	defaultDevice(device as any);
	const cfg = { nLayer: 2, nEmbd: 96, nHead: 4, blockSize: 96, vocab: 24 };
	const data = new Uint16Array(20000);
	for (let i = 0; i < data.length; i++) data[i] = i % cfg.vocab;

	const run = async (syncEvery: number) => {
		let params = gpt.initParams(cfg, 7);
		let s = 1;
		const rand = () => ((s = (s * 1664525 + 1013904223) >>> 0), s / 4294967296);
		const jitStep = jit((p: any, a: any, b: any, t: any) =>
			valueAndGrad((pp: any) => gpt.lossFn(pp, cfg, a, b, t))(p)
		);
		const solver = adam(1e-3);
		let st = solver.init(tree.ref(params));
		// warm up the compile
		{
			const { tokenOH, posOH, targetOH } = gpt.makeBatchOH(cfg, data, rand, 8);
			const [l, g] = jitStep(tree.ref(params), tokenOH, posOH, targetOH);
			const [u, st2] = solver.update(g, st, tree.ref(params));
			params = applyUpdates(params, u);
			st = st2;
			l.item();
		}
		const t0 = performance.now();
		for (let i = 0; i < steps; i++) {
			const { tokenOH, posOH, targetOH } = gpt.makeBatchOH(cfg, data, rand, 8);
			const [l, g] = jitStep(tree.ref(params), tokenOH, posOH, targetOH);
			const [u, st2] = solver.update(g, st, tree.ref(params));
			params = applyUpdates(params, u);
			st = st2;
			if (i % syncEvery === syncEvery - 1) l.item();
			else l.dispose();
			await yieldToPage();
		}
		const { tokenOH, posOH, targetOH } = gpt.makeBatchOH(cfg, data, rand, 8);
		const [l] = jitStep(tree.ref(params), tokenOH, posOH, targetOH);
		l.item(); // final drain
		const ms = (performance.now() - t0) / steps;
		jitStep.dispose();
		gpt.disposeTree(params);
		gpt.disposeTree(st);
		return ms;
	};

	return { every1: await run(1), every10: await run(10) };
}

// ── run ─────────────────────────────────────────────────────────────────────

void (async () => {
	const devices = await init();
	lines.length = 0;
	log(`jax-js devices available: ${devices.join(', ')}`);
	log(`ua: ${navigator.userAgent.match(/Chrom\w+\/[\d.]+/)?.[0] ?? navigator.userAgent.slice(0, 40)}`);
	log();

	log('1 · small MLP [2,64,64,2] on 256 rows — median ms/step');
	for (const d of devices) {
		const ms = await benchMlp(d);
		log(`     ${d.padEnd(7)} ${ms.toFixed(2)} ms`);
	}
	log();

	log('2 · transformer 2×96, 4 heads, block 96, batch 8 — median ms/step');
	for (const d of devices) {
		if (d === 'cpu') {
			log(`     cpu     (skipped — minutes per step)`);
			continue;
		}
		const r = await benchGpt(d);
		log(`     ${d.padEnd(7)} ${r.ms.toFixed(1)} ms   (${r.params.toLocaleString()} params)`);
	}
	log();

	log('3 · optimizer placement, MLP [64,128,128,64] — median ms/step');
	for (const d of devices) {
		if (d === 'cpu') continue;
		const r = await benchOptimizer(d);
		const speedup = ((r.optax / r.fused - 1) * 100).toFixed(0);
		log(
			`     ${d.padEnd(7)} optax-outside ${r.optax.toFixed(2)} ms · fused-inside ${r.fused.toFixed(2)} ms  (${speedup}% faster fused)`
		);
	}
	log();

	log('4 · sync cadence, transformer — mean ms/step (upstream issue #151)');
	for (const d of devices) {
		if (d === 'cpu') continue;
		const r = await benchSync(d);
		log(
			`     ${d.padEnd(7)} sync every step ${r.every1.toFixed(1)} ms · sync every 10 ${r.every10.toFixed(1)} ms`
		);
	}
	log();

	log('5 · transformer: optimizer placement — median ms/step');
	for (const d of devices) {
		if (d === 'cpu' || d === 'webgl') continue;
		const r = await benchGptOptimizer(d);
		log(
			`     ${d.padEnd(7)} optax-outside ${r.optax.toFixed(1)} ms · fused-inside ${r.fused.toFixed(1)} ms  (${((r.optax / r.fused - 1) * 100).toFixed(0)}% faster fused)`
		);
	}
	log();

	log('6 · where does WebGPU overtake wasm? (fused optimizer, batch 8)');
	for (const shape of [
		{ nLayer: 2, nEmbd: 96, nHead: 4, blockSize: 96, vocab: 24 },
		{ nLayer: 4, nEmbd: 192, nHead: 4, blockSize: 128, vocab: 256 },
		{ nLayer: 6, nEmbd: 256, nHead: 8, blockSize: 128, vocab: 1024 }
	]) {
		const n =
			2 * shape.vocab * shape.nEmbd +
			shape.blockSize * shape.nEmbd +
			shape.nLayer * 12 * shape.nEmbd * shape.nEmbd;
		const parts: string[] = [];
		for (const d of ['wasm', 'webgpu']) {
			if (!devices.includes(d as any)) continue;
			parts.push(`${d} ${(await benchGptFused(d, shape)).toFixed(0)} ms`);
		}
		log(`     ${shape.nLayer}×${String(shape.nEmbd).padEnd(4)} V=${String(shape.vocab).padEnd(5)} ${(n / 1e6).toFixed(2)}M params  ·  ${parts.join(' · ')}`);
	}
	log();
	log('done.');
	document.body.dataset.benchDone = '1';
})().catch((e) => log(`error: ${e?.message ?? e}`));
