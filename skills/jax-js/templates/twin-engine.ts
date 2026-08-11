// The twin-worker lab: a trainer that never stops, and a sampler that answers
// the UI from couriered checkpoints.
//
// The problem this solves: if the UI asks the training worker to write a sample
// (or evaluate a position, or render an attention map) while training runs,
// training stops for the duration and the loss curve visibly stutters. With a
// second worker, the trainer keeps stepping on its own GPU device while the
// sampler answers on another.
//
// Guarantees:
//   - the checkpoint is exported AFTER a burst and BEFORE the next train() call,
//     so a sample is exactly the weights of the step it is labelled with
//   - if the sampler fails to boot, or dies mid-courier, sampling silently falls
//     back to the training worker — slower, never broken
//   - a generation counter cancels every in-flight callback on teardown
//
// Framework-agnostic. `notify()` is called whenever observable state changes;
// wire it to $state assignments (Svelte), setState (React), or a store.

import { Engine, detectWebGPU, type ModelConfig, type TrainMetrics } from './engine';
import { toPromptTokens } from './tokens';

export type Phase = 'idle' | 'loading' | 'ready' | 'training' | 'error' | 'no-webgpu';

export interface Sample {
	id: number;
	step: number;
	/** The prompt as the model actually read it — decoded back from its IDs. */
	prompt: string;
	text: string;
}

export interface LabOptions {
	config: ModelConfig;
	tokenData: Uint16Array;
	/** ids → text, for display. Output only; the lab never encodes. */
	decode: (ids: number[]) => string;
	lr?: number;
	/** Steps per burst; a held-out eval and a fresh sample follow each one. */
	chunk?: number;
	/**
	 * The fixed prompt re-asked after every burst, as token IDs. Encode it once
	 * in the application layer with `encodePrompt()` from tokens.ts — this lab
	 * accepts IDs only, so no path exists from free text into model execution.
	 */
	autoPromptTokens?: readonly number[];
	/** Called after any observable field changes. */
	notify?: () => void;
}

/** A stalled fetch or a GPU device that never arrives would otherwise leave the
 *  UI saying "loading…" forever. Give every boot step a deadline. */
function guard<T>(what: string, p: Promise<T>, ms = 25_000): Promise<T> {
	return Promise.race([
		p,
		new Promise<never>((_, reject) =>
			setTimeout(() => reject(new Error(`${what} timed out — try again`)), ms)
		)
	]);
}

export class TwinLab {
	// ── observable state ──────────────────────────────────────────────────────
	phase: Phase = 'idle';
	loadNote = '';
	errorMsg = '';
	step = 0;
	lossNow = NaN;
	tokensPerSec = 0;
	/** [step, loss] per optimizer step. */
	trainLoss: Array<[number, number]> = [];
	/** [step, loss] on fixed held-out batches — step 0, then one per burst. */
	valPoints: Array<[number, number]> = [];
	samples: Sample[] = [];
	paramCount = 0;

	// ── internals: plain fields, never reactive ───────────────────────────────
	private opts: LabOptions;
	private engine: Engine | null = null;
	private sampler: Engine | null = null;
	private samplerReady = false;
	private initCkpt: ArrayBuffer | null = null;
	private webgpu: boolean | null = null;
	private playing = false;
	private trainPromise: Promise<void> | null = null;
	private samplePromise: Promise<void> | null = null;
	private gen = 0;
	private sampleSeq = 0;

	constructor(opts: LabOptions) {
		this.opts = opts;
	}

	private touch() {
		this.opts.notify?.();
	}

	/** Cheap adapter probe on mount, so no-WebGPU users get prose, not a dead
	 *  button. Call this before showing any control. */
	async probe(): Promise<void> {
		if (this.webgpu !== null) return;
		this.webgpu = await detectWebGPU();
		if (!this.webgpu && this.phase === 'idle') this.phase = 'no-webgpu';
		this.touch();
	}

	async boot(): Promise<void> {
		if (this.phase === 'loading' || this.phase === 'ready' || this.phase === 'training') return;
		const myGen = ++this.gen;
		this.errorMsg = '';
		this.phase = 'loading';
		this.touch();
		try {
			if (this.webgpu === null) this.webgpu = await detectWebGPU();
			if (myGen !== this.gen) return;
			if (!this.webgpu) {
				this.phase = 'no-webgpu';
				this.touch();
				return;
			}

			// Awaited, not fired-and-forgotten: the old workers must let go of
			// their GPU devices before the new ones ask for theirs.
			const stale = this.engine;
			const staleSampler = this.sampler;
			this.engine = null;
			this.sampler = null;
			this.samplerReady = false;
			if (stale) await stale.dispose();
			if (staleSampler) await staleSampler.dispose();
			if (myGen !== this.gen) return;

			this.loadNote = 'building the model on your GPU…';
			this.touch();
			const engine = new Engine({
				tokenData: this.opts.tokenData,
				decode: this.opts.decode,
				seed: 42,
				lr: this.opts.lr ?? 1.5e-3
			});
			this.engine = engine;

			// Superseded from here on means: hand the device back. A worker dropped
			// without dispose() keeps its GPU device and the next boot waits forever.
			const superseded = () => {
				if (myGen === this.gen) return false;
				if (this.engine === engine) this.engine = null;
				void engine.dispose();
				return true;
			};

			await guard('the GPU', engine.init(this.opts.config));
			if (superseded()) return;
			this.paramCount = engine.paramCount;

			// Kept so reset() can restore step-0 weights without a re-init.
			this.initCkpt = await guard('the first weights', engine.exportCheckpoint());
			if (superseded()) return;

			const v0 = await engine.valLoss();
			if (superseded()) return;

			this.step = 0;
			this.lossNow = NaN;
			this.trainLoss = [];
			this.valPoints = [[0, v0]];
			this.samples = [];
			this.phase = 'ready';
			this.touch();

			void this.autoSample(); // the step-0 baseline, on record
			void this.bootSampler(myGen); // the second worker, in the background
		} catch (err) {
			if (myGen !== this.gen) return;
			this.errorMsg = err instanceof Error ? err.message : String(err);
			this.phase = 'error';
			this.touch();
		}
	}

	/** Boot the sampling worker. Silent on failure: sampling then rides the
	 *  training worker, which merely brings back the pause it used to cause. */
	private async bootSampler(myGen: number): Promise<void> {
		const s = new Engine({
			tokenData: this.opts.tokenData,
			decode: this.opts.decode,
			seed: 43,
			lr: this.opts.lr ?? 1.5e-3
		});
		try {
			await guard('the sampler', s.init(this.opts.config));
			if (myGen !== this.gen) {
				void s.dispose();
				return;
			}
			this.sampler = s;
			this.samplerReady = true;
		} catch {
			void s.dispose();
		}
	}

	/** Which engine writes the next sample: the sampler, freshly loaded with the
	 *  trainer's current weights (training keeps running), or the training worker
	 *  itself when the sampler is missing (the caller then waits). */
	private async sampleEngine(): Promise<Engine | null> {
		const e = this.engine;
		const s = this.samplerReady ? this.sampler : null;
		if (!e || !s) return e;
		try {
			const ckpt = await e.exportCheckpoint(); // one quick readback
			await s.loadWeights(ckpt); // transferred, not copied
			return s;
		} catch {
			// the sampler died mid-courier — retire it and wait inline again
			this.samplerReady = false;
			this.sampler = null;
			void s.dispose();
			return this.engine;
		}
	}

	// ── transport ─────────────────────────────────────────────────────────────

	toggle(): void {
		if (this.phase === 'training') void this.pause();
		else this.start();
	}

	start(): void {
		if (this.playing || this.phase !== 'ready' || !this.engine) return;
		this.playing = true;
		this.phase = 'training';
		this.touch();
		this.trainPromise = this.loop();
	}

	async pause(): Promise<void> {
		this.playing = false;
		await this.engine?.stop().catch(() => {});
		if (this.trainPromise) await this.trainPromise;
		this.trainPromise = null;
		if (this.phase === 'training') this.phase = 'ready';
		this.touch();
	}

	private async loop(): Promise<void> {
		const myGen = this.gen;
		const chunk = this.opts.chunk ?? 40;
		while (this.playing && this.engine && myGen === this.gen) {
			const e = this.engine;
			try {
				await e.train(chunk, (m: TrainMetrics) => {
					if (myGen !== this.gen) return;
					this.step = m.step;
					this.lossNow = m.loss;
					this.tokensPerSec = this.tokensPerSec
						? this.tokensPerSec * 0.8 + m.tokensPerSec * 0.2
						: m.tokensPerSec;
					this.trainLoss.push([m.step, m.loss]);
					this.touch();
				});
				if (myGen !== this.gen || !this.engine) return;

				const v = await e.valLoss();
				if (myGen !== this.gen) return;
				this.valPoints.push([this.step, v]);
				this.touch();

				// With the sampling worker up, the sample is written on its own
				// device while the next burst runs — the curve never pauses.
				// Without it, wait, as a single-worker loop always had to.
				if (this.samplerReady) void this.autoSample();
				else await this.autoSample();
			} catch (err) {
				if (myGen !== this.gen) return;
				this.playing = false;
				this.errorMsg = err instanceof Error ? err.message : String(err);
				this.phase = 'error';
				this.touch();
				return;
			}
		}
	}

	/** Back to the exact step-0 weights (Adam moments reset with them). */
	async reset(): Promise<void> {
		const e = this.engine;
		if (!e || !this.initCkpt || this.phase === 'loading') return;
		const wasPlaying = this.playing;
		await this.pause();
		// a sample may still be in flight on the sampler; let it land rather than
		// have it surface after the reset
		if (this.samplePromise) await this.samplePromise.catch(() => {});
		const myGen = this.gen;
		try {
			await e.loadWeights(this.initCkpt.slice(0)); // slice: keep our master copy
			if (myGen !== this.gen) return;
			this.step = 0;
			this.lossNow = NaN;
			this.trainLoss = [];
			this.samples = [];
			const v0 = await e.valLoss();
			if (myGen !== this.gen) return;
			this.valPoints = [[0, v0]];
			this.touch();
			await this.autoSample();
			if (myGen !== this.gen) return;
			if (wasPlaying) this.start();
		} catch {
			// engine disposed mid-reset — the gen guard already stopped state writes
		}
	}

	// ── sampling ──────────────────────────────────────────────────────────────

	/** The fixed question, asked every burst: same prompt, same temperature, so
	 *  the only thing that changes between samples is the weights. */
	private autoSample(): Promise<void> {
		const p = this.writeSample(this.opts.autoPromptTokens ?? [], 0.8);
		this.samplePromise = p;
		return p;
	}

	/**
	 * Sample from a caller-supplied prompt, given as token IDs. Encode text with
	 * `encodePrompt()` from tokens.ts before calling — it validates that every ID
	 * is an integer inside the model's vocabulary, and throws if the caller's
	 * encoder and the model disagree.
	 */
	sampleNow(promptTokens: readonly number[], temperature = 0.8): Promise<void> {
		const p = this.writeSample(promptTokens, temperature);
		this.samplePromise = p;
		return p;
	}

	private async writeSample(
		promptTokens: readonly number[],
		temperature: number
	): Promise<void> {
		const myGen = this.gen;
		const atStep = this.step;
		try {
			// Validated at the boundary, and again inside the worker.
			const ids = toPromptTokens(promptTokens as number[], {
				vocab: this.opts.config.vocab,
				maxLen: Math.floor(this.opts.config.blockSize / 2)
			});
			const via = await this.sampleEngine();
			if (!via || myGen !== this.gen) return;
			const r = await via.sample(ids, { temperature, topK: 40, maxTokens: 120 });
			if (myGen !== this.gen) return;
			this.samples = [
				// what the model actually read, decoded back from its own IDs
				{ id: ++this.sampleSeq, step: atStep, prompt: this.opts.decode(ids), text: r.text },
				...this.samples
			].slice(0, 5);
			this.touch();
		} catch {
			// disposed mid-sample, or a prompt whose IDs are not in this vocabulary
		}
	}

	// ── teardown ──────────────────────────────────────────────────────────────

	/** Page unmount: cancel everything in flight, terminate both workers. */
	disposeAll(): void {
		this.gen++; // cancels every in-flight callback
		this.playing = false;
		const e = this.engine;
		const s = this.sampler;
		this.engine = null;
		this.sampler = null;
		this.samplerReady = false;
		if (e) void e.dispose();
		if (s) void s.dispose();
		this.samplePromise = null;
		this.trainPromise = null;
		this.initCkpt = null;
		this.phase = this.webgpu === false ? 'no-webgpu' : 'idle';
		this.step = 0;
		this.lossNow = NaN;
		this.trainLoss = [];
		this.valPoints = [];
		this.samples = [];
		this.touch();
	}
}
