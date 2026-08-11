// Main-thread client for the training worker: a promise RPC with a streaming
// side-channel for per-step metrics. One Engine per mounted model.
//
// This class is a plain object, NOT framework state. Hold it in a module-level
// field or a `let engine: Engine | null = null` — never in $state / useState.

/* eslint-disable @typescript-eslint/no-explicit-any */

import { toPromptTokens } from './tokens';

export interface ModelConfig {
	nLayer: number;
	nEmbd: number;
	nHead: number;
	blockSize: number;
	vocab: number;
}

export interface TrainMetrics {
	step: number;
	loss: number;
	stepMs: number;
	tokensPerSec: number;
}

interface Pending {
	resolve: (v: unknown) => void;
	reject: (e: Error) => void;
	onMetrics?: (m: TrainMetrics) => void;
}

export interface EngineOptions {
	tokenData: Uint16Array;
	seed?: number;
	lr?: number;
	/** ids → text, for display only. Output, never input. */
	decode?: (ids: number[]) => string;
	stopToken?: number;
}

export class Engine {
	private worker: Worker;
	private pending = new Map<number, Pending>();
	private nextId = 1;
	private opts: EngineOptions;
	/** Set at init(); bounds every token sequence the engine forwards. */
	private vocab = 0;
	private blockSize = 0;
	device = 'unknown';
	paramCount = 0;

	constructor(opts: EngineOptions) {
		this.opts = opts;
		// new URL(..., import.meta.url) is what lets Vite/SvelteKit bundle the
		// worker. A string path will not be bundled.
		this.worker = new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' });
		this.worker.onmessage = (e) => this.onMessage(e);
		this.worker.onerror = (e) => {
			const err = new Error(e.message || 'worker error');
			for (const p of this.pending.values()) p.reject(err);
			this.pending.clear();
		};
	}

	private onMessage(e: MessageEvent) {
		const msg = e.data;
		const p = this.pending.get(msg.id);
		if (!p) return;
		if (msg.event === 'metrics') {
			p.onMetrics?.(msg.m as TrainMetrics); // stream — do not settle
			return;
		}
		this.pending.delete(msg.id);
		if (msg.ok) p.resolve(msg.result);
		else p.reject(new Error(msg.error));
	}

	private call<T>(
		op: string,
		payload: Record<string, unknown> = {},
		transfer: Transferable[] = [],
		onMetrics?: (m: TrainMetrics) => void
	): Promise<T> {
		const id = this.nextId++;
		return new Promise<T>((resolve, reject) => {
			this.pending.set(id, { resolve: resolve as (v: unknown) => void, reject, onMetrics });
			this.worker.postMessage({ id, op, ...payload }, transfer);
		});
	}

	async init(config: ModelConfig, checkpoint?: ArrayBuffer): Promise<void> {
		// Copy: the buffer is transferred and would otherwise detach on our side.
		const copy = this.opts.tokenData.slice();
		const payload: Record<string, unknown> = {
			config,
			tokenData: copy.buffer,
			seed: this.opts.seed ?? 42,
			lr: this.opts.lr ?? 3e-4
		};
		const transfer: Transferable[] = [copy.buffer];
		if (checkpoint) {
			payload.checkpoint = checkpoint;
			transfer.push(checkpoint);
		}
		const r = await this.call<{ device: string; paramCount: number }>('init', payload, transfer);
		this.device = r.device;
		this.paramCount = r.paramCount;
		this.vocab = config.vocab;
		this.blockSize = config.blockSize;
	}

	/** Run `steps` updates; metrics stream through onMetrics as they happen. */
	train(steps: number, onMetrics: (m: TrainMetrics) => void): Promise<void> {
		return this.call('train', { steps }, [], onMetrics).then(() => undefined);
	}

	/** Pause an in-flight train() (its promise resolves early). */
	async stop(): Promise<void> {
		await this.call('stop');
	}

	async valLoss(): Promise<number> {
		const r = await this.call<{ valLoss: number }>('valloss');
		return r.valLoss;
	}

	/**
	 * Sample a continuation. `promptTokens` must be integer token IDs, not text:
	 * encode in the application layer with `encodePrompt()` from tokens.ts. The
	 * IDs are validated here and again inside the worker.
	 */
	async sample(
		promptTokens: readonly number[],
		opts?: { temperature?: number; topK?: number; maxTokens?: number }
	): Promise<{ tokens: number[]; text: string }> {
		const r = await this.call<{ tokens: number[] }>('sample', {
			promptTokens: toPromptTokens(promptTokens as number[], {
				vocab: this.vocab,
				maxLen: this.blockSize
			}),
			temperature: opts?.temperature,
			topK: opts?.topK,
			maxTokens: opts?.maxTokens,
			stopToken: this.opts.stopToken
		});
		return { tokens: r.tokens, text: this.opts.decode?.(r.tokens) ?? '' };
	}

	async exportCheckpoint(): Promise<ArrayBuffer> {
		const r = await this.call<{ checkpoint: ArrayBuffer }>('export');
		return r.checkpoint;
	}

	/** Swap the resident weights in place. The buffer is TRANSFERRED — pass a
	 *  copy (`buf.slice(0)`) if you intend to keep it. */
	async loadWeights(checkpoint: ArrayBuffer, opts?: { preserveStep?: boolean }): Promise<void> {
		await this.call('load', { checkpoint, preserveStep: opts?.preserveStep ?? false }, [
			checkpoint
		]);
	}

	/** Swap the training corpus; weights and optimizer survive. */
	async setTokens(tokens: Uint16Array): Promise<number> {
		const copy = tokens.slice();
		const r = await this.call<{ tokens: number }>('settokens', { tokenData: copy.buffer }, [
			copy.buffer
		]);
		return r.tokens;
	}

	/** A worker mid-jit answers no RPC promptly, and a worker that never releases
	 *  its GPU device blocks the next one from getting it — so the graceful
	 *  request gets a deadline, and termination happens either way. */
	async dispose(): Promise<void> {
		try {
			await Promise.race([this.call('dispose'), new Promise((r) => setTimeout(r, 400))]);
		} finally {
			this.worker.terminate();
			this.pending.clear();
		}
	}
}

/** Feature-detect WebGPU. A wedged GPU process answers requestAdapter() never
 *  rather than null, which would leave the UI spinning — treat silence as no. */
export async function detectWebGPU(): Promise<boolean> {
	if (typeof navigator === 'undefined' || !navigator.gpu) return false;
	try {
		const adapter = await Promise.race([
			navigator.gpu.requestAdapter(),
			new Promise<null>((r) => setTimeout(() => r(null), 8000))
		]);
		return adapter !== null;
	} catch {
		return false;
	}
}
