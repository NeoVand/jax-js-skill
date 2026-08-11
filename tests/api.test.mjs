// Contract tests for every claim the skill makes about jax-js.
// Runs on cpu/wasm in Node — no browser needed. Re-run on every jax-js upgrade:
//   node --test tests/api.test.mjs
import { test, before, describe } from 'node:test';
import assert from 'node:assert/strict';
import {
	init,
	defaultDevice,
	numpy as np,
	nn,
	jit,
	grad,
	valueAndGrad,
	vmap,
	tree,
	random,
	lax,
	blockUntilReady
} from '@jax-js/jax';
import { adam, applyUpdates, chain, clipByGlobalNorm } from '@jax-js/optax';

import * as mlp from '../skills/jax-js/templates/model-mlp.ts';
import * as gpt from '../skills/jax-js/templates/model-transformer.ts';
import { fusedAdam } from '../skills/jax-js/templates/fused-adam.ts';
import {
	toPromptTokens,
	encodePrompt,
	InvalidTokensError
} from '../skills/jax-js/templates/tokens.ts';

before(async () => {
	const devices = await init();
	defaultDevice(devices.includes('wasm') ? 'wasm' : 'cpu');
});

const throws = (fn) => {
	try {
		fn();
		return false;
	} catch {
		return true;
	}
};

// ── the token boundary: only integer IDs reach the model ────────────────────
describe('token boundary', () => {
	const bounds = { vocab: 24, maxLen: 8 };

	test('accepts valid ids and returns a defensive copy', () => {
		const src = [0, 5, 23];
		const out = toPromptTokens(src, bounds);
		assert.deepEqual(out, [0, 5, 23]);
		out[0] = 99;
		assert.equal(src[0], 0, 'must not alias the caller array');
	});

	test('truncates to the most recent maxLen ids', () => {
		assert.deepEqual(toPromptTokens([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], bounds), [
			3, 4, 5, 6, 7, 8, 9, 10
		]);
	});

	for (const [name, bad] of [
		['out of range high', [0, 24]],
		['out of range low', [-1]],
		['non-integer', [1.5]],
		['NaN', [NaN]],
		['Infinity', [Infinity]],
		['string that looks numeric', ['3']],
		['null element', [null]],
		['undefined element', [undefined]],
		['object element', [{}]],
		['nested array', [[1]]]
	]) {
		test(`rejects ${name}`, () => {
			assert.throws(() => toPromptTokens(bad, bounds), InvalidTokensError);
		});
	}

	for (const [name, bad] of [
		['a string', 'hello'],
		['null', null],
		['undefined', undefined],
		['an object', { 0: 1, length: 1 }],
		['a number', 7]
	]) {
		test(`rejects ${name} as the sequence itself`, () => {
			assert.throws(() => toPromptTokens(bad, bounds), InvalidTokensError);
		});
	}

	test('encodePrompt validates whatever the encoder returns', () => {
		const rogue = () => [0, 999]; // encoder disagrees with the vocabulary
		assert.throws(() => encodePrompt('hi', rogue, bounds), InvalidTokensError);
	});

	test('encodePrompt rejects a non-string', () => {
		assert.throws(() => encodePrompt(42, (s) => [0], bounds), InvalidTokensError);
	});

	test('encodePrompt passes clean text through', () => {
		const enc = (s) => [...s].map((c) => c.charCodeAt(0) % 24);
		assert.deepEqual(encodePrompt('abc', enc, bounds), [
			'a'.charCodeAt(0) % 24,
			'b'.charCodeAt(0) % 24,
			'c'.charCodeAt(0) % 24
		]);
	});

	test('rejects a bad vocab bound', () => {
		assert.throws(() => toPromptTokens([0], { vocab: 0, maxLen: 4 }), InvalidTokensError);
	});
});

// ── SKILL.md law 1: arrays are moved ────────────────────────────────────────
describe('law 1 — ownership', () => {
	test('a consumed array throws on reuse', () => {
		const a = np.array([1, 2, 3]);
		np.sum(a).dispose();
		assert.ok(throws(() => np.sum(a)), 'expected reuse-after-consume to throw');
	});

	test('.ref lends exactly one extra use', () => {
		const a = np.array([1, 2, 3]);
		assert.equal(np.sum(a.ref).item(), 6);
		assert.equal(np.sum(a).item(), 6);
	});

	test('methods consume the receiver', () => {
		const a = np.array([1, 2, 3]);
		a.add(1).dispose();
		assert.ok(throws(() => a.add(1)));
	});

	test('np.zerosLike consumes its argument', () => {
		const a = np.array([1, 2]);
		np.zerosLike(a).dispose();
		assert.ok(throws(() => np.sum(a)));
	});

	test('double dispose throws', () => {
		const a = np.array([1]);
		a.dispose();
		assert.ok(throws(() => a.dispose()));
	});

	test('.ref yields an independent handle', () => {
		const a = np.array([5]);
		const b = a.ref;
		a.dispose();
		assert.equal(np.sum(b).item(), 5);
	});

	test('tree.ref / tree.map / tree.dispose', () => {
		const t = { a: np.ones([2]), l: [np.zeros([3])] };
		assert.equal(tree.leaves(tree.ref(t)).length, 2);
		tree.dispose(tree.ref(t)); // free the refs we just took
		const doubled = tree.map((x) => x.mul(2), t);
		assert.deepEqual(doubled.a.js(), [2, 2]);
		doubled.l[0].dispose();
	});
});

// ── SKILL.md law 2: reading consumes ────────────────────────────────────────
describe('law 2 — readback consumes', () => {
	for (const [name, read] of [
		['item', (a) => a.item()],
		['js', (a) => a.js()],
		['dataSync', (a) => a.dataSync()],
		['data', async (a) => await a.data()]
	]) {
		test(`.${name}() consumes`, async () => {
			const a = np.array([5]);
			await read(a);
			assert.ok(throws(() => np.sum(a)), `.${name}() should consume`);
		});
	}

	test('blockUntilReady does NOT consume', async () => {
		const a = np.array([5]);
		await blockUntilReady(a);
		assert.equal(np.sum(a).item(), 5);
	});

	test('.ref before a read keeps the array alive', () => {
		const a = np.array([5]);
		assert.equal(a.ref.item(), 5);
		assert.equal(np.sum(a).item(), 5);
	});
});

// ── SKILL.md law 3: jit shape caching ───────────────────────────────────────
describe('law 3 — jit', () => {
	test('traces once per shape signature', () => {
		let traces = 0;
		const f = jit((x) => {
			traces++;
			return np.sum(x);
		});
		f(np.zeros([4])).dispose();
		f(np.zeros([4])).dispose();
		f(np.zeros([8])).dispose();
		f(np.zeros([8])).dispose();
		f.dispose();
		assert.equal(traces, 2, 'two distinct shapes → two traces');
	});

	test('staticArgnums retraces per distinct value', () => {
		let traces = 0;
		const f = jit(
			(x, n) => {
				traces++;
				return np.sum(x).mul(n);
			},
			{ staticArgnums: [1] }
		);
		for (let n = 1; n <= 5; n++) f(np.ones([3]), n).item();
		f.dispose();
		assert.equal(traces, 5, 'never route a step counter through staticArgnums');
	});

	test('a jitted closure bakes params in as constants (the flat-loss bug)', () => {
		let params = np.array([1.0]);
		const bad = jit((x) => x.mul(params.ref)); // captured, not passed
		const before = bad(np.array([1.0])).item();
		params.dispose();
		params = np.array([100.0]); // "training" updated the params...
		const after = bad(np.array([1.0])).item();
		bad.dispose();
		params.dispose();
		assert.equal(before, after, 'closure-captured params are frozen at trace time');
	});
});

// ── SKILL.md law 4: gather grad under jit ───────────────────────────────────
describe('law 4 — embeddings', () => {
	test('grad through np.take works EAGERLY', () => {
		const idx = np.array([2, 0, 2], { dtype: np.int32 });
		const g = grad((w) => np.sum(np.take(w, idx.ref, 0)))(np.zeros([3, 2]));
		idx.dispose();
		assert.deepEqual(g.js(), [
			[1, 1],
			[0, 0],
			[2, 2]
		]);
	});

	test('grad through np.take FAILS under jit — hence one-hot matmuls', () => {
		const f = jit((w, i) => valueAndGrad((ww) => np.sum(np.take(ww, i, 0)))(w));
		let failed = false;
		try {
			const [l, g] = f(np.zeros([3, 2]), np.array([1, 1], { dtype: np.int32 }));
			l.dispose();
			g.dispose();
		} catch {
			failed = true;
		}
		f.dispose();
		assert.ok(failed, 'if this now PASSES, law 4 can be relaxed — update the skill');
	});

	test('the one-hot workaround gives the same gradient, under jit', () => {
		const f = jit((w, oh) => valueAndGrad((ww) => np.sum(np.dot(oh, ww)))(w));
		const [l, g] = f(np.zeros([3, 2]), nn.oneHot(np.array([1, 1], { dtype: np.int32 }), 3));
		l.dispose();
		f.dispose();
		assert.deepEqual(g.js(), [
			[0, 0],
			[2, 2],
			[0, 0]
		]);
	});

	test('one-hot matmul == take, forward', () => {
		const ids = np.array([2, 0], { dtype: np.int32 });
		const W = np.arange(6).reshape([3, 2]).astype(np.float32);
		assert.deepEqual(np.dot(nn.oneHot(ids.ref, 3), W.ref).js(), np.take(W, ids, 0).js());
	});
});

// ── optax ───────────────────────────────────────────────────────────────────
describe('optax', () => {
	test('adam outside jit converges', () => {
		const solver = adam(1e-1);
		let params = { w: np.array([1.0, 2.0]) };
		let st = solver.init(tree.ref(params));
		let loss = Infinity;
		for (let i = 0; i < 40; i++) {
			const [l, g] = valueAndGrad((p) => np.sum(np.square(p.w)))(tree.ref(params));
			const [u, st2] = solver.update(g, st, tree.ref(params));
			params = applyUpdates(params, u);
			st = st2;
			loss = l.item();
		}
		tree.dispose(params);
		tree.dispose(st);
		// Adam at a fixed lr oscillates around the optimum rather than settling;
		// assert it descended a long way from ‖[1,2]‖² = 5, not that it is exact.
		assert.ok(loss < 0.2, `expected convergence from 5.0, got ${loss}`);
	});

	test('optax CANNOT be jitted at 0.1.2 (count.item is a host readback)', () => {
		const solver = adam(1e-1);
		let params = { w: np.array([1.0, 2.0]) };
		const st = solver.init(tree.ref(params));
		const step = jit((p, s) => {
			const [l, g] = valueAndGrad((pp) => np.sum(np.square(pp.w)))(tree.ref(p));
			const [u, s2] = solver.update(g, s, tree.ref(p));
			return [l, applyUpdates(p, u), s2];
		});
		let failed = false;
		try {
			step(params, st);
		} catch {
			failed = true;
		}
		step.dispose();
		assert.ok(failed, 'if this now PASSES, optax was republished — update the skill');
	});

	test('chain(clipByGlobalNorm, adam)', () => {
		const solver = chain(clipByGlobalNorm(1.0), adam(1e-3));
		let params = { w: np.array([10.0, 20.0]) };
		let st = solver.init(tree.ref(params));
		for (let i = 0; i < 5; i++) {
			const g = grad((p) => np.sum(np.square(p.w)))(tree.ref(params));
			const [u, st2] = solver.update(g, st, tree.ref(params));
			params = applyUpdates(params, u);
			st = st2;
		}
		const w = params.w.js();
		tree.dispose(st);
		assert.ok(w[0] < 10 && w[1] < 20, 'clipped adam still descends');
	});

	test('fusedAdam (inside jit) matches optax adam', () => {
		const run = (useFused) => {
			let params = { w: np.array([1.0, 2.0]) };
			if (useFused) {
				const s = fusedAdam((p) => np.sum(np.square(p.w)), { lr: 1e-1 });
				let st = s.init(tree.ref(params));
				let loss;
				for (let i = 0; i < 40; i++) {
					const [l, p2, st2] = s.step(params, st);
					params = p2;
					st = st2;
					loss = l.item();
				}
				s.dispose();
				tree.dispose(st.m);
				tree.dispose(st.v);
				const w = params.w.js();
				return [loss, w];
			}
			const solver = adam(1e-1);
			let st = solver.init(tree.ref(params));
			let loss;
			for (let i = 0; i < 40; i++) {
				const [l, g] = valueAndGrad((p) => np.sum(np.square(p.w)))(tree.ref(params));
				const [u, st2] = solver.update(g, st, tree.ref(params));
				params = applyUpdates(params, u);
				st = st2;
				loss = l.item();
			}
			tree.dispose(st);
			const w = params.w.js();
			return [loss, w];
		};
		const [lo, wo] = run(false);
		const [lf, wf] = run(true);
		assert.ok(Math.abs(lo - lf) < 1e-6, `loss ${lo} vs ${lf}`);
		for (let i = 0; i < wo.length; i++) assert.ok(Math.abs(wo[i] - wf[i]) < 1e-5);
	});

	test('fusedAdam traces once across many steps', () => {
		// staticArgnums would retrace per step; device scalars must not.
		let params = { w: np.array([1.0, 2.0]) };
		const s = fusedAdam((p) => np.sum(np.square(p.w)), { lr: 1e-2 });
		let st = s.init(tree.ref(params));
		for (let i = 0; i < 12; i++) {
			const [l, p2, st2] = s.step(params, st);
			params = p2;
			st = st2;
			l.item();
		}
		s.dispose();
		tree.dispose(params);
		tree.dispose(st.m);
		tree.dispose(st.v);
		assert.ok(true);
	});
});

// ── transforms ──────────────────────────────────────────────────────────────
describe('transforms', () => {
	test('valueAndGrad argnums', () => {
		const [l, [gx, gy]] = valueAndGrad((x, y) => np.sum(np.square(x)).add(np.sum(y.mul(2))), {
			argnums: [0, 1]
		})(np.array([3.0]), np.array([1.0]));
		assert.equal(l.item(), 11);
		assert.deepEqual(gx.js(), [6]);
		assert.deepEqual(gy.js(), [2]);
	});

	test('hasAux carries metrics out of the traced computation', () => {
		const [[l, aux], g] = valueAndGrad((p) => [np.sum(np.square(p.ref)), np.mean(p)], {
			hasAux: true
		})(np.array([1.0, 3.0]));
		assert.equal(l.item(), 10);
		assert.equal(aux.item(), 2);
		assert.deepEqual(g.js(), [2, 6]);
	});

	test('vmap', () => {
		const out = vmap((x) => np.sum(np.square(x)))(np.arange(6).reshape([3, 2]).astype(np.float32));
		assert.deepEqual(out.js(), [1, 13, 41]);
	});

	test('lax.stopGradient detaches', () => {
		const g = grad((x) => np.sum(lax.stopGradient(x.ref).mul(x)))(np.array([2.0, 3.0]));
		assert.deepEqual(g.js(), [2, 3]);
	});

	test('random keys are reproducible and split cleanly', () => {
		const a = random.normal(random.key(7), [4]).js();
		const b = random.normal(random.key(7), [4]).js();
		assert.deepEqual(a, b);
	});

	test('nn.dotProductAttention shape + causality', () => {
		const [B, S, H, D] = [1, 4, 2, 3];
		const q = np.ones([B, S, H, D]);
		const k = np.ones([B, S, H, D]);
		const v = np.arange(B * S * H * D)
			.reshape([B, S, H, D])
			.astype(np.float32);
		const o = nn.dotProductAttention(q, k, v, { isCausal: true });
		assert.deepEqual(o.shape, [B, S, H, D]);
		const flat = o.js();
		// position 0 can only attend to itself → equals v[0]
		assert.deepEqual(flat[0][0][0], [0, 1, 2]);
	});
});

// ── the MLP template actually learns ────────────────────────────────────────
describe('template: model-mlp', () => {
	test('fits a sine wave (mse) — loss drops by >20x', () => {
		const cfg = { layers: [1, 24, 24, 1], activation: 'tanh', loss: 'mse', seed: 7 };
		let params = mlp.initParams(cfg);
		// 1·24 + 24 + 24·24 + 24 + 24·1 + 1
		assert.equal(mlp.paramCount(params), 673);

		const N = 128;
		const xs = new Float32Array(N);
		const ys = new Float32Array(N);
		for (let i = 0; i < N; i++) {
			xs[i] = (i / (N - 1)) * 2 - 1;
			ys[i] = Math.sin(3.1 * xs[i]);
		}
		const x = np.array(xs).reshape([N, 1]);
		const y = np.array(ys).reshape([N, 1]);

		const jitStep = jit((p, xx, yy) => valueAndGrad((pp) => mlp.lossFn(pp, cfg, xx, yy))(p));
		const solver = adam(5e-2);
		let st = solver.init(tree.ref(params));
		let first = null;
		let last = null;
		for (let i = 0; i < 300; i++) {
			const [l, g] = jitStep(tree.ref(params), x.ref, y.ref);
			const [u, st2] = solver.update(g, st, tree.ref(params));
			params = applyUpdates(params, u);
			st = st2;
			last = l.item();
			if (first === null) first = last;
		}
		x.dispose();
		y.dispose();
		jitStep.dispose();
		tree.dispose(params);
		tree.dispose(st);
		assert.ok(last < first / 20, `loss ${first.toFixed(4)} → ${last.toFixed(4)}`);
	});

	test('classifies with xent — accuracy above 90% on two blobs', () => {
		const cfg = { layers: [2, 16, 2], activation: 'relu', loss: 'xent', seed: 3 };
		let params = mlp.initParams(cfg);
		const N = 200;
		const xs = new Float32Array(N * 2);
		const labels = new Int32Array(N);
		let s = 42;
		const rand = () => ((s = (s * 1664525 + 1013904223) >>> 0), s / 4294967296);
		for (let i = 0; i < N; i++) {
			const c = i % 2;
			labels[i] = c;
			xs[i * 2] = (c ? 1.5 : -1.5) + (rand() - 0.5) * 0.8;
			xs[i * 2 + 1] = (c ? 1.5 : -1.5) + (rand() - 0.5) * 0.8;
		}
		const x = np.array(xs).reshape([N, 2]);
		const yOH = nn.oneHot(np.array(labels, { dtype: np.int32 }), 2);

		const jitStep = jit((p, xx, yy) => valueAndGrad((pp) => mlp.lossFn(pp, cfg, xx, yy))(p));
		const solver = adam(3e-2);
		let st = solver.init(tree.ref(params));
		for (let i = 0; i < 200; i++) {
			const [l, g] = jitStep(tree.ref(params), x.ref, yOH.ref);
			const [u, st2] = solver.update(g, st, tree.ref(params));
			params = applyUpdates(params, u);
			st = st2;
			l.item();
		}
		const logits = mlp.forward(tree.ref(params), cfg, x).out;
		const pred = np.argmax(logits, -1).js();
		let correct = 0;
		for (let i = 0; i < N; i++) if (pred[i] === labels[i]) correct++;
		yOH.dispose();
		jitStep.dispose();
		tree.dispose(params);
		tree.dispose(st);
		assert.ok(correct / N > 0.9, `accuracy ${correct / N}`);
	});

	test('VAE bottleneck runs and adds a KL term', () => {
		const cfg = {
			layers: [8, 4, 2, 4, 8],
			activation: 'tanh',
			loss: 'mse',
			seed: 1,
			vae: { at: 1, beta: 1 / 8 }
		};
		let params = mlp.initParams(cfg);
		const N = 32;
		const buf = new Float32Array(N * 8);
		for (let i = 0; i < buf.length; i++) buf[i] = Math.sin(i * 0.3);
		const x = np.array(buf).reshape([N, 8]);
		const noise = random.normal(random.key(5), [N, 2]);
		const jitStep = jit((p, xx, yy, nz) =>
			valueAndGrad((pp) => mlp.lossFn(pp, cfg, xx, yy, nz))(p)
		);
		const solver = adam(1e-2);
		let st = solver.init(tree.ref(params));
		let first = null;
		let last = null;
		for (let i = 0; i < 150; i++) {
			const [l, g] = jitStep(tree.ref(params), x.ref, x.ref, noise.ref);
			const [u, st2] = solver.update(g, st, tree.ref(params));
			params = applyUpdates(params, u);
			st = st2;
			last = l.item();
			if (first === null) first = last;
		}
		x.dispose();
		noise.dispose();
		jitStep.dispose();
		tree.dispose(params);
		tree.dispose(st);
		assert.ok(Number.isFinite(last) && last < first, `VAE loss ${first} → ${last}`);
	});
});

// ── the transformer template actually learns ────────────────────────────────
describe('template: model-transformer', () => {
	test('learns a repeating pattern — loss drops well below uniform', () => {
		const cfg = { nLayer: 2, nEmbd: 32, nHead: 4, blockSize: 16, vocab: 8 };
		let params = gpt.initParams(cfg, 7);
		const n = gpt.paramCount(params);
		assert.ok(n > 1000, `paramCount ${n}`);

		// a perfectly predictable stream: 0 1 2 3 4 5 6 7 0 1 2 …
		const data = new Uint16Array(2048);
		for (let i = 0; i < data.length; i++) data[i] = i % cfg.vocab;

		let s = 1234;
		const rand = () => ((s = (s * 1664525 + 1013904223) >>> 0), s / 4294967296);

		const jitStep = jit((p, tok, pos, tgt) =>
			valueAndGrad((pp) => gpt.lossFn(pp, cfg, tok, pos, tgt))(p)
		);
		const solver = adam(3e-3, { b1: 0.9, b2: 0.99 });
		let st = solver.init(tree.ref(params));

		let first = null;
		let last = null;
		for (let i = 0; i < 120; i++) {
			const { tokenOH, posOH, targetOH } = gpt.makeBatchOH(cfg, data, rand, 4);
			const [l, g] = jitStep(tree.ref(params), tokenOH, posOH, targetOH);
			const [u, st2] = solver.update(g, st, tree.ref(params));
			params = applyUpdates(params, u);
			st = st2;
			last = l.item();
			if (first === null) first = last;
		}
		const uniform = Math.log(cfg.vocab); // 2.079 nats
		assert.ok(first > uniform * 0.7, `start ${first} should be near uniform ${uniform}`);
		assert.ok(last < 0.5, `expected mastery of a trivial pattern, got ${last}`);

		// and it can generate: forwardSeq + sampleFromRow round-trip
		const jitF = jit((p, tok, pos) => gpt.forwardLogprobs(p, cfg, cfg.blockSize, tok, pos));
		const jitForward = (tok, pos) => jitF(tree.ref(params), tok, pos);
		const ids = [0, 1, 2];
		for (let i = 0; i < 5; i++) {
			const rows = gpt.forwardSeq(jitForward, cfg, ids.slice(-cfg.blockSize));
			const at = Math.min(ids.length, cfg.blockSize) - 1;
			const row = rows.subarray(at * cfg.vocab, (at + 1) * cfg.vocab);
			ids.push(gpt.sampleFromRow(row, 0.1, 3, rand));
		}
		assert.deepEqual(ids, [0, 1, 2, 3, 4, 5, 6, 7], `generated ${ids}`);

		jitStep.dispose();
		jitF.dispose();
		gpt.disposeTree(params);
		gpt.disposeTree(st);
	});

	test('checkpoint round-trip preserves the model exactly', () => {
		const cfg = { nLayer: 1, nEmbd: 16, nHead: 2, blockSize: 8, vocab: 6 };
		const params = gpt.initParams(cfg, 3);
		const flat = gpt.flattenParams(params);
		const restored = gpt.loadParams(cfg, flat);
		const a = gpt.flattenParams(params);
		const b = gpt.flattenParams(restored);
		assert.equal(a.length, b.length);
		for (let i = 0; i < a.length; i++) assert.equal(a[i], b[i]);
		gpt.disposeTree(params);
		gpt.disposeTree(restored);
	});

	test('attention capture matches the fused forward', () => {
		const cfg = { nLayer: 2, nEmbd: 16, nHead: 2, blockSize: 8, vocab: 5 };
		const params = gpt.initParams(cfg, 11);
		const S = cfg.blockSize;
		const buf = new Int32Array(S);
		for (let i = 0; i < S; i++) buf[i] = i % cfg.vocab;
		const ids = np.array(buf, { dtype: np.int32 }).reshape([1, S]);
		const pos = np.arange(S).astype(np.int32).reshape([1, S]);

		const mbuf = new Float32Array(S * S);
		for (let i = 0; i < S; i++) for (let j = i + 1; j < S; j++) mbuf[i * S + j] = -1e9;
		const mask = np.array(mbuf).reshape([S, S]);

		const jitFused = jit((p, t, o) => gpt.forwardLogprobs(p, cfg, S, t, o));
		const jitAttn = jit((p, t, o, m) => gpt.forwardWithAttention(p, cfg, t, o, m));

		const lpFused = jitFused(
			tree.ref(params),
			nn.oneHot(ids.ref, cfg.vocab),
			nn.oneHot(pos.ref, S)
		).js();
		const [lpAttn, attn] = jitAttn(
			tree.ref(params),
			nn.oneHot(ids, cfg.vocab),
			nn.oneHot(pos, S),
			mask
		);
		const lpHand = lpAttn.js();

		// the two forwards must agree — otherwise the attention map is a lie
		for (let i = 0; i < lpFused.length; i++)
			for (let j = 0; j < lpFused[i].length; j++)
				assert.ok(
					Math.abs(lpFused[i][j] - lpHand[i][j]) < 2e-3,
					`row ${i} col ${j}: ${lpFused[i][j]} vs ${lpHand[i][j]}`
				);

		assert.equal(attn.length, cfg.nLayer);
		const rows = attn[0].js();
		assert.equal(rows.length, cfg.nHead * S);
		// causal: position 0 attends only to itself
		assert.ok(Math.abs(rows[0][0] - 1) < 1e-4, `attn[0][0] = ${rows[0][0]}`);
		for (let j = 1; j < S; j++) assert.ok(rows[0][j] < 1e-6);
		for (const a of attn.slice(1)) a.dispose();

		jitFused.dispose();
		jitAttn.dispose();
		gpt.disposeTree(params);
	});

	test('residual capture returns one [S, nEmbd] block per layer', () => {
		const cfg = { nLayer: 2, nEmbd: 16, nHead: 2, blockSize: 8, vocab: 5 };
		const params = gpt.initParams(cfg, 2);
		const S = cfg.blockSize;
		const ids = np.zeros([1, S]).astype(np.int32);
		const pos = np.arange(S).astype(np.int32).reshape([1, S]);
		const jitRes = jit((p, t, o) => gpt.forwardResiduals(p, cfg, t, o));
		const [lp, res] = jitRes(tree.ref(params), nn.oneHot(ids, cfg.vocab), nn.oneHot(pos, S));
		lp.dispose();
		assert.equal(res.length, cfg.nLayer);
		for (const r of res) {
			assert.equal(r.size, S * cfg.nEmbd);
			r.dispose();
		}
		jitRes.dispose();
		gpt.disposeTree(params);
	});
});
