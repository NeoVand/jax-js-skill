// The training worker: owns jax-js, the params, the optimizer state and the
// data. The main thread drives it over the tiny RPC in engine.ts.
//
// Everything heavy lives here. The main thread never sees a jax-js array.
// Adapt the `handlers` table; the dispatch, transfer and stop machinery below
// is the part you should not need to change.

import {
	init,
	defaultDevice,
	numpy as np,
	nn,
	jit,
	valueAndGrad,
	tree,
	blockUntilReady
} from '@jax-js/jax';
import { adam, applyUpdates } from '@jax-js/optax';
import * as model from './model-transformer';
import type { ModelConfig } from './model-transformer';
import { toPromptTokens } from './tokens';

/* eslint-disable @typescript-eslint/no-explicit-any */

interface RpcRequest {
	id: number;
	op: string;
	[key: string]: unknown;
}

const post = (msg: unknown, transfer?: Transferable[]) =>
	(self as unknown as Worker).postMessage(msg, { transfer: transfer ?? [] });

// ── worker state ────────────────────────────────────────────────────────────
let cfg: ModelConfig | null = null;
let params: any = null;
let optState: any = null;
let solver: ReturnType<typeof adam> | null = null;
let tokenData: Uint16Array | null = null;
let jitStep: any = null;
let jitForward: any = null;
let jitLoss: any = null;
let valStart = 0;
let stepCounter = 0;
let stopRequested = false;
let device = 'none';

const BATCH = 8;

function mulberry32(seed: number) {
	return function () {
		seed |= 0;
		seed = (seed + 0x6d2b79f5) | 0;
		let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
		t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
		return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
	};
}
let rng = mulberry32(1234);

/** One optimizer step. The loss is synced every step on purpose: letting steps
 *  queue up and blocking once at the end measures ~2× SLOWER on WebGPU. */
function trainStep(): number {
	const c = cfg!;
	const { tokenOH, posOH, targetOH } = model.makeBatchOH(c, tokenData!, rng, BATCH, 0, valStart);
	const [lossVal, grads] = jitStep(tree.ref(params), tokenOH.ref, posOH.ref, targetOH.ref);
	const [updates, nextState] = solver!.update(grads, optState, tree.ref(params));
	params = applyUpdates(params, updates);
	optState = nextState;
	tokenOH.dispose();
	posOH.dispose();
	targetOH.dispose();
	return lossVal.item(); // consumes lossVal — do not dispose after
}

// ── op handlers ─────────────────────────────────────────────────────────────

async function handleInit(req: RpcRequest) {
	const devices = await init();
	// Prefer WebGPU; fall back rather than dying, unless the model is too big.
	device = devices.includes('webgpu') ? 'webgpu' : devices.includes('wasm') ? 'wasm' : 'cpu';
	defaultDevice(device as any);

	cfg = req.config as ModelConfig;
	tokenData = new Uint16Array(req.tokenData as ArrayBuffer);
	rng = mulberry32((req.seed as number) ?? 1234);
	stepCounter = 0;

	if (params) model.disposeTree(params);
	if (optState) model.disposeTree(optState);
	params = req.checkpoint
		? model.loadParams(cfg, new Float32Array(req.checkpoint as ArrayBuffer))
		: model.initParams(cfg, (req.seed as number) ?? 42);
	await blockUntilReady(params);

	solver = adam((req.lr as number) ?? 3e-4, { b1: 0.9, b2: 0.99 });
	optState = solver.init(tree.ref(params));

	const c = cfg;
	// params flow in as ARGUMENTS — a closure would bake step-0 weights in.
	jitStep = jit((p: any, a: any, b: any, t: any) =>
		valueAndGrad((pp: any) => model.lossFn(pp, c, a, b, t))(p)
	);
	jitForward = jit((p: any, a: any, b: any) => model.forwardLogprobs(p, c, c.blockSize, a, b));
	jitLoss = jit((p: any, a: any, b: any, t: any) => model.lossFn(p, c, a, b, t));

	// held-out tail: train from [0, valStart), validate from the rest
	valStart = Math.floor(tokenData.length * 0.95);
	const minVal = 4 * (c.blockSize + 1);
	if (tokenData.length - valStart < minVal) valStart = Math.max(1, tokenData.length - minVal);

	return { device, paramCount: model.paramCount(params), tokens: tokenData.length };
}

async function handleTrain(req: RpcRequest) {
	const steps = (req.steps as number) ?? 50;
	stopRequested = false;
	let done = 0;
	for (let i = 0; i < steps; i++) {
		if (stopRequested) break;
		const t0 = performance.now();
		const loss = trainStep();
		const stepMs = performance.now() - t0;
		stepCounter++;
		done++;
		post({
			id: req.id,
			event: 'metrics',
			m: {
				step: stepCounter,
				loss,
				stepMs,
				tokensPerSec: Math.round((BATCH * cfg!.blockSize * 1000) / stepMs)
			}
		});
		// Yield to the worker's own event loop so 'stop' messages get through.
		if (i % 4 === 3) await new Promise((r) => setTimeout(r, 0));
	}
	return { completed: done, step: stepCounter };
}

/** Mean loss over a few FIXED held-out batches (deterministic seed, so calls
 *  across training are comparable — the curve is real validation loss). */
function handleValLoss() {
	const c = cfg!;
	const vr = mulberry32(9999);
	const N = 4;
	let total = 0;
	for (let i = 0; i < N; i++) {
		const { tokenOH, posOH, targetOH } = model.makeBatchOH(
			c,
			tokenData!,
			vr,
			BATCH,
			valStart,
			tokenData!.length
		);
		const lossVal = jitLoss(tree.ref(params), tokenOH.ref, posOH.ref, targetOH.ref);
		tokenOH.dispose();
		posOH.dispose();
		targetOH.dispose();
		total += lossVal.item();
	}
	return { valLoss: total / N };
}

/** Bound a numeric RPC field to a sane range; the worker trusts no input. */
function num(v: unknown, fallback: number, lo: number, hi: number): number {
	const n = typeof v === 'number' && Number.isFinite(v) ? v : fallback;
	return Math.min(hi, Math.max(lo, n));
}

function handleSample(req: RpcRequest) {
	const c = cfg!;
	// The message contract is integer token IDs only — never text. Re-validated
	// here because the main thread is not a trust boundary (see tokens.ts).
	const prompt = toPromptTokens(req.promptTokens ?? [], {
		vocab: c.vocab,
		maxLen: Math.floor(c.blockSize / 2)
	});
	const temperature = num(req.temperature, 0.8, 1e-4, 100);
	const topK = Math.floor(num(req.topK, 40, 0, c.vocab));
	const maxTokens = Math.floor(num(req.maxTokens, 120, 1, c.blockSize - 1));
	const stopAt =
		typeof req.stopToken === 'number' && Number.isInteger(req.stopToken) ? req.stopToken : undefined;

	const forward = (tok: any, pos: any) => jitForward(tree.ref(params), tok, pos);
	let tokens = prompt.slice(-Math.floor(c.blockSize / 2));
	const generated: number[] = [];
	for (let i = 0; i < maxTokens; i++) {
		const lp = model.forwardSeq(forward, c, tokens);
		const at = Math.min(tokens.length, c.blockSize) - 1;
		const row = lp.subarray(at * c.vocab, (at + 1) * c.vocab) as Float32Array;
		const next = model.sampleFromRow(row, temperature, topK, rng);
		if (stopAt !== undefined && next === stopAt) break;
		generated.push(next);
		tokens.push(next);
		if (tokens.length >= c.blockSize) tokens = tokens.slice(-Math.floor(c.blockSize / 2));
	}
	return { tokens: generated };
}

/** Serialise the weights. Returned as a transferable — no copy. */
function handleExport() {
	const flat = model.flattenParams(params);
	return { checkpoint: flat.buffer, __transfer: [flat.buffer] };
}

/** Replace the resident weights in place. Far cheaper than a re-init: the
 *  config, the jit caches and the dataset all survive, so this costs one buffer
 *  transfer. This is what the twin-worker courier calls. */
async function handleLoad(req: RpcRequest) {
	if (!cfg || !solver) throw new Error('load before init');
	const flat = new Float32Array(req.checkpoint as ArrayBuffer);
	if (params) model.disposeTree(params);
	params = model.loadParams(cfg, flat);
	await blockUntilReady(params);
	if (optState) model.disposeTree(optState);
	optState = solver.init(tree.ref(params)); // a snapshot carries no Adam history
	if (!req.preserveStep) stepCounter = 0;
	return { paramCount: model.paramCount(params) };
}

/** Swap the training corpus in place — params, optimizer state and jits all
 *  survive. Continuing training on a curated corpus IS supervised fine-tuning. */
function handleSetTokens(req: RpcRequest) {
	tokenData = new Uint16Array(req.tokenData as ArrayBuffer);
	valStart = Math.floor(tokenData.length * 0.95);
	return { tokens: tokenData.length };
}

function handleDispose() {
	if (params) model.disposeTree(params);
	if (optState) model.disposeTree(optState);
	jitStep?.dispose();
	jitForward?.dispose();
	jitLoss?.dispose();
	params = optState = jitStep = jitForward = jitLoss = null;
	return {};
}

// ── dispatch ────────────────────────────────────────────────────────────────
const handlers: Record<string, (req: RpcRequest) => unknown | Promise<unknown>> = {
	init: handleInit,
	train: handleTrain,
	stop: () => {
		stopRequested = true;
		return {};
	},
	valloss: handleValLoss,
	sample: handleSample,
	export: handleExport,
	load: handleLoad,
	settokens: handleSetTokens,
	dispose: handleDispose
};

self.onmessage = async (e: MessageEvent<RpcRequest>) => {
	const req = e.data;
	try {
		const handler = handlers[req.op];
		if (!handler) throw new Error(`unknown op: ${req.op}`);
		const result = (await handler(req)) as Record<string, unknown> & {
			__transfer?: Transferable[];
		};
		const transfer = result?.__transfer;
		if (transfer) delete result.__transfer;
		post({ id: req.id, ok: true, result }, transfer);
	} catch (err) {
		post({
			id: req.id,
			ok: false,
			error: err instanceof Error ? `${err.name}: ${err.message}` : String(err)
		});
	}
};
