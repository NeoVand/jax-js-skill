#!/usr/bin/env node
// Verify that the jax-js installed in THIS project still behaves the way the
// skill assumes. Run it before trusting the guidance, and again after any
// upgrade — jax-js moves fast, and two of the five laws exist only because of
// current gaps that may close.
//
//   node scripts/doctor.mjs
//
// Exit code 0 = every assumption holds. 1 = something changed; the output says
// which law is affected.

import { createRequire } from 'node:module';
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';

const require = createRequire(`${process.cwd()}/package.json`);

/** The package's `exports` map hides package.json, so resolve the entry point
 *  and walk up until we find one. */
function versionOf(pkg) {
	try {
		let dir = dirname(require.resolve(pkg, { paths: [process.cwd()] }));
		for (let i = 0; i < 5; i++) {
			try {
				return JSON.parse(readFileSync(join(dir, 'package.json'), 'utf8')).version;
			} catch {
				dir = dirname(dir);
			}
		}
	} catch {
		/* not installed */
	}
	return null;
}

let jax, optax;
try {
	jax = await import(require.resolve('@jax-js/jax', { paths: [process.cwd()] }));
} catch {
	console.error('✗ @jax-js/jax is not installed in this project.\n  npm i @jax-js/jax @jax-js/optax');
	process.exit(1);
}
try {
	optax = await import(require.resolve('@jax-js/optax', { paths: [process.cwd()] }));
} catch {
	optax = null;
}

const { init, defaultDevice, numpy: np, nn, jit, grad, valueAndGrad, tree, blockUntilReady } = jax;

const jaxVersion = versionOf('@jax-js/jax') ?? 'unknown';
const optaxVersion = optax ? versionOf('@jax-js/optax') : null;

const KNOWN_JAX = '0.1.21';
const KNOWN_OPTAX = '0.1.2';

const results = [];
const check = async (law, name, fn) => {
	try {
		const detail = await fn();
		results.push({ ok: true, law, name, detail: detail ?? '' });
	} catch (e) {
		results.push({ ok: false, law, name, detail: `${e?.name}: ${e?.message}` });
	}
};
const assert = (cond, msg) => {
	if (!cond) throw new Error(msg);
};
const consumes = (fn) => {
	const a = np.array([5]);
	fn(a);
	try {
		np.sum(a).dispose();
		return false;
	} catch {
		return true;
	}
};

const devices = await init();
defaultDevice(devices.includes('wasm') ? 'wasm' : 'cpu');

console.log(`jax-js        ${jaxVersion}${jaxVersion === KNOWN_JAX ? '' : `  (skill written against ${KNOWN_JAX})`}`);
console.log(`optax         ${optaxVersion ?? 'not installed'}${!optaxVersion || optaxVersion === KNOWN_OPTAX ? '' : `  (skill written against ${KNOWN_OPTAX})`}`);
console.log(`devices       ${devices.join(', ')}`);
console.log(`default       ${defaultDevice()}`);
console.log('');

// ── law 1 ───────────────────────────────────────────────────────────────────
await check(1, 'arrays are moved: reuse after consume throws', () => {
	const a = np.array([1, 2, 3]);
	np.sum(a).dispose();
	let threw = false;
	try {
		np.sum(a);
	} catch {
		threw = true;
	}
	assert(threw, 'reuse after consume did NOT throw — ownership rules may have changed');
	return 'move semantics active';
});

await check(1, '.ref lends exactly one extra use', () => {
	const a = np.array([1, 2, 3]);
	assert(np.sum(a.ref).item() === 6, 'bad sum');
	assert(np.sum(a).item() === 6, 'bad sum');
	return 'ok';
});

// ── law 2 ───────────────────────────────────────────────────────────────────
await check(2, 'item()/js()/dataSync()/data() all consume', async () => {
	const which = [];
	if (consumes((a) => a.item())) which.push('item');
	if (consumes((a) => a.js())) which.push('js');
	if (consumes((a) => a.dataSync())) which.push('dataSync');
	const a = np.array([5]);
	await a.data();
	let dataConsumes = false;
	try {
		np.sum(a).dispose();
	} catch {
		dataConsumes = true;
	}
	if (dataConsumes) which.push('data');
	assert(which.length === 4, `only ${which.join(', ')} consume — law 2 needs updating`);
	return 'all four readbacks consume';
});

await check(2, 'blockUntilReady does not consume', async () => {
	const a = np.array([5]);
	await blockUntilReady(a);
	assert(np.sum(a).item() === 5, 'blockUntilReady consumed its argument');
	return 'ok';
});

// ── law 3 ───────────────────────────────────────────────────────────────────
await check(3, 'jit traces once per shape signature', () => {
	let traces = 0;
	const f = jit((x) => {
		traces++;
		return np.sum(x);
	});
	f(np.zeros([4])).dispose();
	f(np.zeros([4])).dispose();
	f(np.zeros([8])).dispose();
	f.dispose();
	assert(traces === 2, `expected 2 traces, got ${traces}`);
	return 'shape-keyed cache confirmed';
});

await check(3, 'staticArgnums retraces per distinct value', () => {
	let traces = 0;
	const f = jit(
		(x, n) => {
			traces++;
			return np.sum(x).mul(n);
		},
		{ staticArgnums: [1] }
	);
	for (let n = 1; n <= 4; n++) f(np.ones([3]), n).item();
	f.dispose();
	assert(traces === 4, `expected 4 traces, got ${traces}`);
	return 'never pass a step counter through staticArgnums';
});

// ── law 4 ───────────────────────────────────────────────────────────────────
await check(4, 'grad through np.take still FAILS under jit', () => {
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
	assert(
		failed,
		'GOOD NEWS: jitted gather backward now works. Law 4 can be relaxed — ' +
			'embeddings may use np.take instead of one-hot matmuls, which saves B·S·V floats.'
	);
	return 'one-hot embeddings still required';
});

await check(4, 'the one-hot workaround gives correct gradients under jit', () => {
	const f = jit((w, oh) => valueAndGrad((ww) => np.sum(np.dot(oh, ww)))(w));
	const [l, g] = f(np.zeros([3, 2]), nn.oneHot(np.array([1, 1], { dtype: np.int32 }), 3));
	l.dispose();
	f.dispose();
	assert(JSON.stringify(g.js()) === '[[0,0],[2,2],[0,0]]', 'wrong gradient');
	return 'ok';
});

// ── optax ───────────────────────────────────────────────────────────────────
if (optax) {
	await check('optax', 'optax still cannot be placed inside jit', () => {
		const solver = optax.adam(1e-1);
		const params = { w: np.array([1.0, 2.0]) };
		const st = solver.init(tree.ref(params));
		const step = jit((p, s) => {
			const [l, g] = valueAndGrad((pp) => np.sum(np.square(pp.w)))(tree.ref(p));
			const [u, s2] = solver.update(g, s, tree.ref(p));
			return [l, optax.applyUpdates(p, u), s2];
		});
		let failed = false;
		try {
			step(params, st);
		} catch {
			failed = true;
		}
		step.dispose();
		assert(
			failed,
			'GOOD NEWS: optax can now be jitted. templates/fused-adam.ts is no longer ' +
				'needed — put solver.update inside the jitted step.'
		);
		return 'keep solver.update outside jit (or use templates/fused-adam.ts)';
	});

	await check('optax', 'adam converges outside jit', () => {
		const solver = optax.adam(1e-1);
		let params = { w: np.array([1.0, 2.0]) };
		let st = solver.init(tree.ref(params));
		let loss = Infinity;
		for (let i = 0; i < 40; i++) {
			const [l, g] = valueAndGrad((p) => np.sum(np.square(p.w)))(tree.ref(params));
			const [u, st2] = solver.update(g, st, tree.ref(params));
			params = optax.applyUpdates(params, u);
			st = st2;
			loss = l.item();
		}
		tree.dispose(params);
		tree.dispose(st);
		assert(loss < 0.2, `loss ${loss} — expected descent from 5.0`);
		return `loss 5.0 → ${loss.toExponential(2)}`;
	});
}

// ── a whole model trains ────────────────────────────────────────────────────
if (optax) {
	await check('end-to-end', 'a 2-layer MLP actually learns', () => {
		const rand = (() => {
			let s = 7;
			return () => ((s = (s * 1664525 + 1013904223) >>> 0), s / 4294967296);
		})();
		const N = 64;
		const xs = new Float32Array(N);
		const ys = new Float32Array(N);
		for (let i = 0; i < N; i++) {
			xs[i] = (i / (N - 1)) * 2 - 1;
			ys[i] = Math.sin(3 * xs[i]);
		}
		const x = np.array(xs).reshape([N, 1]);
		const y = np.array(ys).reshape([N, 1]);
		const mk = (fin, fout) => {
			const b = new Float32Array(fin * fout);
			const lim = Math.sqrt(6 / (fin + fout));
			for (let i = 0; i < b.length; i++) b[i] = (rand() * 2 - 1) * lim;
			return np.array(b).reshape([fin, fout]);
		};
		let params = { w: [mk(1, 24), mk(24, 1)], b: [np.zeros([24]), np.zeros([1])] };
		const fwd = (p, xx) => np.dot(np.tanh(np.dot(xx, p.w[0]).add(p.b[0])), p.w[1]).add(p.b[1]);
		const step = jit((p, xx, yy) =>
			valueAndGrad((pp) => np.mean(np.square(fwd(pp, xx).sub(yy))))(p)
		);
		const solver = optax.adam(5e-2);
		let st = solver.init(tree.ref(params));
		let first = null;
		let last = null;
		for (let i = 0; i < 200; i++) {
			const [l, g] = step(tree.ref(params), x.ref, y.ref);
			const [u, st2] = solver.update(g, st, tree.ref(params));
			params = optax.applyUpdates(params, u);
			st = st2;
			last = l.item();
			if (first === null) first = last;
		}
		x.dispose();
		y.dispose();
		step.dispose();
		tree.dispose(params);
		tree.dispose(st);
		assert(last < first / 10, `loss ${first.toFixed(4)} → ${last.toFixed(4)} — not learning`);
		return `loss ${first.toFixed(4)} → ${last.toFixed(4)}`;
	});
}

// ── report ──────────────────────────────────────────────────────────────────
let failed = 0;
for (const r of results) {
	if (!r.ok) failed++;
	const tag = typeof r.law === 'number' ? `law ${r.law}` : r.law;
	console.log(`${r.ok ? '✓' : '✗'} [${tag}] ${r.name}`);
	if (r.detail) console.log(`     ${r.detail}`);
}
console.log('');
if (failed === 0) {
	console.log(`All ${results.length} assumptions hold on jax-js ${jaxVersion}.`);
	process.exit(0);
}
console.log(`${failed} of ${results.length} assumptions changed.`);
console.log('Read the messages above: some failures are GOOD NEWS (a workaround is no');
console.log('longer needed) and mean the skill can be simplified.');
process.exit(1);
