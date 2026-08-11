// One small, general MLP that covers most in-browser demos: curve fitting (mse),
// 2-D classification (xent), MNIST (xent), autoencoders (mse against the input),
// and VAEs (a Gaussian bottleneck).
//
// Ownership: every function consumes what it is handed. See references/memory.md.

import { numpy as np, nn, random, tree } from '@jax-js/jax';

/* eslint-disable @typescript-eslint/no-explicit-any */

export type Activation = 'tanh' | 'relu' | 'gelu' | 'silu';
export type LossKind = 'mse' | 'xent';

export interface MlpConfig {
	/** Layer widths, input first, output last — e.g. [2, 16, 16, 2]. */
	layers: number[];
	activation: Activation;
	loss: LossKind;
	seed?: number;
	/**
	 * Turn one layer into a variational bottleneck. Layer `at` then emits
	 * 2 × layers[at + 1] numbers — a mean and a log-variance — and the tail
	 * receives a sample of that Gaussian while training, or the mean on every
	 * read-out path. `beta` weights KL(q ‖ N(0, I)) against reconstruction;
	 * because the loss averages over elements, textbook β = 1 is `1 / outDim`.
	 * The waist is always linear: a non-linearity there fights the prior.
	 */
	vae?: { at: number; beta: number };
}

export interface MlpParams {
	w: any[];
	b: any[];
}

const act = (kind: Activation, x: any) =>
	kind === 'relu' ? nn.relu(x) : kind === 'gelu' ? nn.gelu(x) : kind === 'silu' ? nn.silu(x) : np.tanh(x);

/** Glorot-uniform weights, zero biases. Hidden layers may be zero-biased safely;
 *  output *weights* must not be zero (that would kill the gradient path). */
export function initParams(cfg: MlpConfig, seed = cfg.seed ?? 0): MlpParams {
	const shapes = layerShapes(cfg);
	const n = shapes.length;
	const keys = random.split(random.key(seed), Math.max(2, n));
	const w: any[] = [];
	const b: any[] = [];
	for (let i = 0; i < n; i++) {
		const [fin, fout] = shapes[i];
		const limit = Math.sqrt(6 / (fin + fout));
		const k = i < n - 1 ? keys.ref.slice(i) : keys.slice(i);
		w.push(random.uniform(k, [fin, fout], { minval: -limit, maxval: limit }));
		b.push(np.zeros([fout]));
	}
	return { w, b };
}

/**
 * [fin, fout] per weight matrix. Inputs always come from cfg.layers — only the
 * waist layer's OUTPUT doubles, because it emits a mean and a log-variance and
 * the next layer still receives a latent of the declared width.
 */
function layerShapes(cfg: MlpConfig): Array<[number, number]> {
	const out: Array<[number, number]> = [];
	for (let i = 0; i < cfg.layers.length - 1; i++) {
		const fout = cfg.vae && i === cfg.vae.at ? 2 * cfg.layers[i + 1] : cfg.layers[i + 1];
		out.push([cfg.layers[i], fout]);
	}
	return out;
}

/**
 * Forward pass. Returns logits (xent) or values (mse).
 * With `vae`, also returns the mean/log-variance so the loss can add the KL
 * term; pass `noise` (a [n, latent] sample) to reparameterise, or null to use
 * the mean (every read-out path: predict, eval, activations).
 */
export function forward(
	p: MlpParams,
	cfg: MlpConfig,
	x: any,
	noise: any = null
): { out: any; mu: any | null; logvar: any | null } {
	let h = x;
	let mu: any = null;
	let logvar: any = null;
	const n = p.w.length;
	for (let k = 0; k < n; k++) {
		h = np.dot(h, p.w[k]).add(p.b[k]);
		if (cfg.vae && k === cfg.vae.at) {
			const half = cfg.layers[cfg.vae.at + 1];
			mu = h.ref.slice([], [0, half]);
			logvar = h.slice([], [half, 2 * half]);
			// reparameterise: z = mu + exp(logvar/2) · ε
			h = noise
				? mu.ref.add(np.exp(logvar.ref.mul(0.5)).mul(noise))
				: mu.ref;
			continue; // the waist stays linear
		}
		if (k < n - 1) h = act(cfg.activation, h);
	}
	return { out: h, mu, logvar };
}

/**
 * Loss. `y` is a Float32 target matrix for 'mse' or a one-hot [n, classes]
 * matrix for 'xent' (build the one-hot outside the jitted step — law 4).
 */
export function lossFn(p: MlpParams, cfg: MlpConfig, x: any, y: any, noise: any = null): any {
	const { out, mu, logvar } = forward(p, cfg, x, noise);
	let loss: any;
	if (cfg.loss === 'xent') {
		const logp = nn.logSoftmax(out, -1);
		loss = np.mean(np.sum(logp.mul(y), -1).neg());
	} else {
		loss = np.mean(np.square(out.sub(y)));
	}
	if (cfg.vae && mu && logvar) {
		// KL(N(mu, σ²) ‖ N(0, I)) = ½ Σ (μ² + σ² − 1 − log σ²)
		const kl = np
			.mean(
				np.sum(
					np.square(mu).add(np.exp(logvar.ref)).sub(1).sub(logvar),
					-1
				)
			)
			.mul(0.5);
		loss = loss.add(kl.mul(cfg.vae.beta));
	} else {
		if (mu) mu.dispose();
		if (logvar) logvar.dispose();
	}
	return loss;
}

/** Post-activation values of every hidden layer — what a "network inspector"
 *  visualisation draws. Inference only. */
export function activations(p: MlpParams, cfg: MlpConfig, x: any): any[] {
	const out: any[] = [];
	let h = x;
	const n = p.w.length;
	for (let k = 0; k < n; k++) {
		h = np.dot(h, p.w[k]).add(p.b[k]);
		if (cfg.vae && k === cfg.vae.at) {
			const half = cfg.layers[cfg.vae.at + 1];
			h = h.slice([], [0, half]); // the mean, on read-out paths
		} else if (k < n - 1) {
			h = act(cfg.activation, h);
		}
		out.push(h.ref);
	}
	h.dispose();
	return out;
}

export function paramCount(p: MlpParams): number {
	const leaves = tree.leaves(tree.ref(p)) as any[];
	const total = leaves.reduce((s, l) => s + l.size, 0);
	for (const l of leaves) l.dispose();
	return total;
}
