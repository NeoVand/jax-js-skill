// A decoder-only transformer small enough to train live in a browser tab.
//
// Idioms that matter here, all of them load-bearing:
//   - every projection is a 2-D matmul on [B·S, D]; the batch is folded into rows
//   - embeddings are one-hot matmuls (grad through np.take throws inside jit)
//   - output projections (wo, mlpFc2, lmHead) are initialised SMALL-RANDOM, not
//     zero: zero-init blocks all gradient into the attention/MLP interiors at
//     step 0 and the model never starts learning
//   - pre-norm RMSNorm, no biases, no dropout — the smallest thing that trains
//   - forward() is called under jit with params as an ARGUMENT, never a closure
//
// Ownership: every function here consumes the arrays it is given exactly once.
// Pass `.ref` if the caller still needs them.

import { numpy as np, nn, random, tree } from '@jax-js/jax';

/* eslint-disable @typescript-eslint/no-explicit-any */

export interface ModelConfig {
	nLayer: number;
	nEmbd: number;
	nHead: number;
	blockSize: number;
	vocab: number;
}

export interface ModelParams {
	wte: any; // [V, D] token embedding
	wpe: any; // [S, D] learned positional embedding
	lmHead: any; // [D, V] output projection
	layers: Array<{ wq: any; wk: any; wv: any; wo: any; mlpFc1: any; mlpFc2: any }>;
}

/** Scaled-uniform init. The 0.2× / 0.4× factors on the output projections keep
 *  the residual stream from exploding while still admitting gradient. */
export function initParams(cfg: ModelConfig, seed: number): ModelParams {
	const s = Math.sqrt(3 / cfg.nEmbd);
	const n = 3 + cfg.nLayer * 6;
	const keys = random.split(random.key(seed), n);
	let ki = 0;
	const nk = () => {
		ki++;
		return ki < n ? keys.ref.slice(ki - 1) : keys.slice(ki - 1);
	};
	const params: ModelParams = {
		wte: random.normal(nk(), [cfg.vocab, cfg.nEmbd]).mul(0.02),
		wpe: random.normal(nk(), [cfg.blockSize, cfg.nEmbd]).mul(0.02),
		lmHead: random.normal(nk(), [cfg.nEmbd, cfg.vocab]).mul(0.001),
		layers: []
	};
	for (let i = 0; i < cfg.nLayer; i++) {
		params.layers.push({
			wq: random.uniform(nk(), [cfg.nEmbd, cfg.nEmbd], { minval: -s, maxval: s }),
			wk: random.uniform(nk(), [cfg.nEmbd, cfg.nEmbd], { minval: -s, maxval: s }),
			wv: random.uniform(nk(), [cfg.nEmbd, cfg.nEmbd], { minval: -s, maxval: s }),
			wo: random.uniform(nk(), [cfg.nEmbd, cfg.nEmbd], { minval: -0.2 * s, maxval: 0.2 * s }),
			mlpFc1: random.uniform(nk(), [cfg.nEmbd, 4 * cfg.nEmbd], {
				minval: -0.4 * s,
				maxval: 0.4 * s
			}),
			mlpFc2: random.uniform(nk(), [4 * cfg.nEmbd, cfg.nEmbd], {
				minval: -0.2 * s,
				maxval: 0.2 * s
			})
		});
	}
	return params;
}

function rmsnorm(x: any) {
	const ms = np.mean(np.square(x.ref), -1, { keepdims: true });
	return x.div(np.sqrt(ms.add(1e-5)));
}

/**
 * Forward to log-probs over the whole sequence: [B·S, vocab].
 * `seqLen` is the length the one-hots were built at (always cfg.blockSize if you
 * pad, which you should — see the note on jit shape caching in SKILL.md law 3).
 * Consumes tokenOH, posOH, and every params leaf exactly once.
 */
export function forwardLogprobs(
	params: any,
	cfg: ModelConfig,
	seqLen: number,
	tokenOH: any,
	posOH: any
) {
	const headDim = cfg.nEmbd / cfg.nHead;
	let x = np.dot(tokenOH.reshape([-1, cfg.vocab]), params.wte);
	const posEmb = np.dot(posOH.reshape([-1, cfg.blockSize]), params.wpe);
	x = rmsnorm(x.add(posEmb));
	for (let li = 0; li < cfg.nLayer; li++) {
		const layer = params.layers[li];
		const xRes = x.ref;
		x = rmsnorm(x);
		const q = np.dot(x.ref, layer.wq);
		const k = np.dot(x.ref, layer.wk);
		const v = np.dot(x, layer.wv);
		// dotProductAttention wants [B, S, H, headDim]
		const qH = q.reshape([-1, seqLen, cfg.nHead, headDim]);
		const kH = k.reshape([-1, seqLen, cfg.nHead, headDim]);
		const vH = v.reshape([-1, seqLen, cfg.nHead, headDim]);
		const attnOut = nn.dotProductAttention(qH, kH, vH, { isCausal: true });
		x = np.dot(attnOut.reshape([-1, cfg.nEmbd]), layer.wo).add(xRes);
		const mlpRes = x.ref;
		x = rmsnorm(x);
		x = nn.relu(np.dot(x, layer.mlpFc1));
		x = np.dot(x, layer.mlpFc2).add(mlpRes);
	}
	const logits = np.dot(x, params.lmHead);
	return nn.logSoftmax(logits, -1);
}

/** Mean next-token NLL. Consumes all three one-hots. */
export function lossFn(params: any, cfg: ModelConfig, tokenOH: any, posOH: any, targetOH: any) {
	const logprobs = forwardLogprobs(params, cfg, cfg.blockSize, tokenOH, posOH);
	return np.mean(np.sum(logprobs.mul(targetOH.reshape([-1, cfg.vocab])), -1).neg());
}

/**
 * Same numerics as forwardLogprobs, but ALSO returns every attention pattern:
 * per layer, an [H·S, S] block of softmax(QKᵀ/√d + mask) rows. Attention is
 * computed by hand because the fused op never materialises the weights.
 * Inference only (jit without grad); B = 1; S is always cfg.blockSize.
 * `causalMask` is [S, S] with 0 on/below the diagonal and -1e9 above; the
 * reference passed in is CONSUMED (pass mask.ref to keep your master copy).
 */
export function forwardWithAttention(
	params: any,
	cfg: ModelConfig,
	tokenOH: any,
	posOH: any,
	causalMask: any
): [any, any[]] {
	const H = cfg.nHead;
	const headDim = cfg.nEmbd / H;
	const scale = 1 / Math.sqrt(headDim);
	let x = np.dot(tokenOH.reshape([-1, cfg.vocab]), params.wte);
	const posEmb = np.dot(posOH.reshape([-1, cfg.blockSize]), params.wpe);
	x = rmsnorm(x.add(posEmb));
	const attnPerLayer: any[] = [];
	for (let li = 0; li < cfg.nLayer; li++) {
		const layer = params.layers[li];
		const xRes = x.ref;
		x = rmsnorm(x);
		const q = np.dot(x.ref, layer.wq);
		const k = np.dot(x.ref, layer.wk);
		const v = np.dot(x, layer.wv);
		const weights: any[] = [];
		const outs: any[] = [];
		for (let h = 0; h < H; h++) {
			const cols: [number, number] = [h * headDim, (h + 1) * headDim];
			const qh = (h < H - 1 ? q.ref : q).slice([], cols);
			const kh = (h < H - 1 ? k.ref : k).slice([], cols);
			const vh = (h < H - 1 ? v.ref : v).slice([], cols);
			const lastUse = li === cfg.nLayer - 1 && h === H - 1;
			const scores = np
				.dot(qh, kh.transpose())
				.mul(scale)
				.add(lastUse ? causalMask : causalMask.ref);
			const w = nn.softmax(scores, -1);
			weights.push(w.ref);
			outs.push(np.dot(w, vh));
		}
		attnPerLayer.push(np.concatenate(weights, 0)); // [H·S, S]
		x = np.dot(np.concatenate(outs, 1), layer.wo).add(xRes);
		const mlpRes = x.ref;
		x = rmsnorm(x);
		x = nn.relu(np.dot(x, layer.mlpFc1));
		x = np.dot(x, layer.mlpFc2).add(mlpRes);
	}
	return [nn.logSoftmax(np.dot(x, params.lmHead), -1), attnPerLayer];
}

/**
 * Same numerics again, returning the residual stream after every block:
 * nLayer tensors of [S, nEmbd]. This is the surface a linear probe reads to ask
 * whether the model has built an internal model of anything.
 */
export function forwardResiduals(
	params: any,
	cfg: ModelConfig,
	tokenOH: any,
	posOH: any
): [any, any[]] {
	const headDim = cfg.nEmbd / cfg.nHead;
	let x = np.dot(tokenOH.reshape([-1, cfg.vocab]), params.wte);
	const posEmb = np.dot(posOH.reshape([-1, cfg.blockSize]), params.wpe);
	x = rmsnorm(x.add(posEmb));
	const resPerLayer: any[] = [];
	for (let li = 0; li < cfg.nLayer; li++) {
		const layer = params.layers[li];
		const xRes = x.ref;
		x = rmsnorm(x);
		const q = np.dot(x.ref, layer.wq);
		const k = np.dot(x.ref, layer.wk);
		const v = np.dot(x, layer.wv);
		const qH = q.reshape([-1, cfg.blockSize, cfg.nHead, headDim]);
		const kH = k.reshape([-1, cfg.blockSize, cfg.nHead, headDim]);
		const vH = v.reshape([-1, cfg.blockSize, cfg.nHead, headDim]);
		const attnOut = nn.dotProductAttention(qH, kH, vH, { isCausal: true });
		x = np.dot(attnOut.reshape([-1, cfg.nEmbd]), layer.wo).add(xRes);
		const mlpRes = x.ref;
		x = rmsnorm(x);
		x = nn.relu(np.dot(x, layer.mlpFc1));
		x = np.dot(x, layer.mlpFc2).add(mlpRes);
		resPerLayer.push(x.ref);
	}
	return [nn.logSoftmax(np.dot(x, params.lmHead), -1), resPerLayer];
}

// ── batching ────────────────────────────────────────────────────────────────

/** Sample `batch` windows of blockSize+1 tokens from a flat token stream and
 *  return them as one-hots. Build these OUTSIDE the jitted step. */
export function makeBatchOH(
	cfg: ModelConfig,
	data: Uint16Array,
	rand: () => number,
	batch: number,
	lo = 0,
	hi = data.length
) {
	const S = cfg.blockSize;
	const inputBuf = new Int32Array(batch * S);
	const targetBuf = new Int32Array(batch * S);
	const span = Math.max(1, Math.min(hi, data.length - S - 1) - lo);
	for (let b = 0; b < batch; b++) {
		const start = lo + Math.floor(rand() * span);
		for (let i = 0; i < S; i++) {
			inputBuf[b * S + i] = data[start + i];
			targetBuf[b * S + i] = data[start + i + 1];
		}
	}
	const inputIds = np.array(inputBuf, { dtype: np.int32 }).reshape([batch, S]);
	const posIds = np.tile(np.arange(S).astype(np.int32), [batch, 1]);
	const targetIds = np.array(targetBuf, { dtype: np.int32 }).reshape([batch, S]);
	return {
		tokenOH: nn.oneHot(inputIds, cfg.vocab),
		posOH: nn.oneHot(posIds, cfg.blockSize),
		targetOH: nn.oneHot(targetIds, cfg.vocab)
	};
}

/**
 * Fixed-shape single-sequence forward: tokens right-padded to blockSize.
 * Causal attention makes padding after the last real position irrelevant, so
 * ONE jit signature serves every prompt length — no per-length recompiles.
 * `jitForward` must be jit((p, tok, pos) => forwardLogprobs(p, cfg, S, tok, pos))
 * with params already bound by the caller via tree.ref.
 */
export function forwardSeq(
	jitForward: (tok: any, pos: any) => any,
	cfg: ModelConfig,
	tokens: number[]
): Float32Array {
	const S = cfg.blockSize;
	const buf = new Int32Array(S);
	for (let i = 0; i < Math.min(tokens.length, S); i++) buf[i] = tokens[i];
	const inputIds = np.array(buf, { dtype: np.int32 }).reshape([1, S]);
	const posIds = np.arange(S).astype(np.int32).reshape([1, S]);
	const logprobs = jitForward(nn.oneHot(inputIds, cfg.vocab), nn.oneHot(posIds, cfg.blockSize));
	return logprobs.dataSync() as Float32Array;
}

/** Temperature + top-k draw from one row of log-probs, on the host. */
export function sampleFromRow(
	row: Float32Array,
	temperature: number,
	topK: number,
	rand: () => number
): number {
	const V = row.length;
	let idx = Array.from({ length: V }, (_, i) => i);
	if (topK > 0 && topK < V) {
		idx.sort((a, b) => row[b] - row[a]);
		idx = idx.slice(0, topK);
	}
	const t = Math.max(temperature, 1e-4);
	let maxv = -Infinity;
	for (const i of idx) maxv = Math.max(maxv, row[i] / t);
	let sum = 0;
	const ps = idx.map((i) => {
		const p = Math.exp(row[i] / t - maxv);
		sum += p;
		return p;
	});
	let r = rand() * sum;
	for (let j = 0; j < idx.length; j++) {
		r -= ps[j];
		if (r <= 0) return idx[j];
	}
	return idx[idx.length - 1];
}

// ── checkpoints ─────────────────────────────────────────────────────────────

export function paramCount(params: ModelParams): number {
	const leaves = tree.leaves(tree.ref(params)) as any[];
	const total = leaves.reduce((s: number, l: any) => s + l.size, 0);
	for (const l of leaves) l.dispose();
	return total;
}

export function disposeTree(t: any): void {
	for (const leaf of tree.leaves(t)) leaf.dispose();
}

/** Flatten to one Float32Array — the checkpoint the courier transfers. */
export function flattenParams(params: ModelParams): Float32Array {
	const leaves = tree.leaves(tree.ref(params)) as any[];
	const total = leaves.reduce((s: number, l: any) => s + l.size, 0);
	const out = new Float32Array(total);
	let offset = 0;
	for (const leaf of leaves) {
		out.set(leaf.dataSync(), offset);
		offset += leaf.size;
	}
	return out;
}

/** Rebuild a params tree from a flat buffer, using cfg for shapes. */
export function loadParams(cfg: ModelConfig, flat: Float32Array): ModelParams {
	const template = initParams(cfg, 0);
	let offset = 0;
	return tree.map((leaf: any) => {
		const chunk = flat.slice(offset, offset + leaf.size);
		offset += leaf.size;
		const next = np.array(chunk).reshape(leaf.shape);
		leaf.dispose();
		return next;
	}, template) as unknown as ModelParams;
}
