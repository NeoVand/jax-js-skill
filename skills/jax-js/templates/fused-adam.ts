// Adam fused into the jitted step — one dispatch per training step, no per-step
// JavaScript in the optimizer at all.
//
// Why this exists: @jax-js/optax@0.1.2 cannot be placed inside jit, because its
// treeBiasCorrection() calls count.item(), a host readback (`TypeError:
// count.item is not a function` when traced). Keeping the optimizer outside jit
// is fine for most models; this is for when the step is small enough that the
// JS round-trip is a real fraction of the time.
//
// Verified to converge bit-for-bit with optax's adam() on the same problem.
//
// The bias-correction constants go in as DEVICE SCALARS, not staticArgnums:
// a static argument recompiles for every distinct value, and the step counter
// changes every step, so staticArgnums would retrace the whole model each time.

import { numpy as np, jit, valueAndGrad, tree, type JsTree } from '@jax-js/jax';

/* eslint-disable @typescript-eslint/no-explicit-any */

export interface AdamOpts {
	lr?: number;
	b1?: number;
	b2?: number;
	eps?: number;
}

export interface FusedAdam<P> {
	/** Run one step. Returns [loss, params, state]; all inputs are consumed. */
	step(params: P, state: AdamState, ...args: any[]): [any, P, AdamState];
	init(params: P): AdamState;
	dispose(): void;
}

export interface AdamState {
	m: any;
	v: any;
	t: number;
}

/**
 * Build a fused Adam step for `lossFn(params, ...args) -> scalar`.
 * Extra args (batches, masks) are passed through to the loss untouched and are
 * consumed by the jitted call, exactly like any other array argument.
 */
export function fusedAdam<P extends JsTree<any>>(
	lossFn: (params: P, ...args: any[]) => any,
	{ lr = 1e-3, b1 = 0.9, b2 = 0.999, eps = 1e-8 }: AdamOpts = {}
): FusedAdam<P> {
	const jitted = jit((p: any, m: any, v: any, c1: any, c2: any, ...args: any[]) => {
		const [loss, g] = valueAndGrad((pp: any) => lossFn(pp, ...args))(tree.ref(p));
		const m2 = tree.map((mm: any, gg: any) => mm.mul(b1).add(gg.mul(1 - b1)), m, tree.ref(g));
		const v2 = tree.map((vv: any, gg: any) => vv.mul(b2).add(np.square(gg).mul(1 - b2)), v, g);
		const upd = tree.map(
			(mm: any, vv: any) => mm.div(c1.ref).div(np.sqrt(vv.div(c2.ref)).add(eps)),
			tree.ref(m2),
			tree.ref(v2)
		);
		c1.dispose();
		c2.dispose();
		const p2 = tree.map((pp: any, uu: any) => pp.sub(uu.mul(lr)), p, upd);
		return [loss, p2, m2, v2];
	});

	return {
		init(params: P): AdamState {
			return {
				m: tree.map((x: any) => np.zerosLike(x), tree.ref(params)),
				v: tree.map((x: any) => np.zerosLike(x), tree.ref(params)),
				t: 0
			};
		},
		step(params: P, state: AdamState, ...args: any[]): [any, P, AdamState] {
			const t = state.t + 1;
			const [loss, p2, m2, v2] = jitted(
				params,
				state.m,
				state.v,
				np.array(1 - Math.pow(b1, t)),
				np.array(1 - Math.pow(b2, t)),
				...args
			) as any;
			return [loss, p2 as P, { m: m2, v: v2, t }];
		},
		dispose() {
			jitted.dispose();
		}
	};
}

/* Usage:

const solver = fusedAdam((p, x, y) => lossFn(p, cfg, x, y), { lr: 3e-4 });
let state = solver.init(params);

for (let i = 0; i < steps; i++) {
  const { x, y } = nextBatch();
  const [lossVal, p2, s2] = solver.step(params, state, x, y);
  params = p2; state = s2;
  loss = lossVal.item();            // consumes — do not dispose after
}

solver.dispose();
tree.dispose(params);
tree.dispose(state.m); tree.dispose(state.v);

*/
