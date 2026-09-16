// Main-thread client for the training worker: a promise RPC with a streaming
// side-channel for per-step metrics. One Engine per mounted model.
//
// Keep the resource handle in a component-owned field or ref, and expose
// metrics as reactive state. Dispose even if initialization is still pending.

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
	private closed = false;
	private disposal: Promise<void> | null = null;
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
