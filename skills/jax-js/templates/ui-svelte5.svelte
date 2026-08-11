<!--
  A training plate in Svelte 5 runes: lifecycle, phases, a live loss curve and a
  transport that cannot get out of sync with the model.

  The two rules this file exists to demonstrate:
    1. the engine is a PLAIN field, never $state — it holds a Worker, and a
       reactive proxy breaks postMessage
    2. the demo boots when it scrolls into view, not on mount — a prerendered
       page should not spin up a GPU device the reader may never reach
-->
<script lang="ts">
	import { onDestroy } from 'svelte';
	import { Engine, detectWebGPU, type ModelConfig, type TrainMetrics } from '$lib/engine';
	import { inview } from '$lib/inview';

	interface Props {
		config: ModelConfig;
		tokenData: Uint16Array;
		/** Steps per burst; an eval follows each one. Fixed — no speed control. */
		chunk?: number;
	}
	let { config, tokenData, chunk = 40 }: Props = $props();

	type Phase = 'idle' | 'loading' | 'ready' | 'training' | 'error' | 'no-webgpu';

	let phase = $state<Phase>('idle');
	let loadNote = $state('');
	let errorMsg = $state('');
	let step = $state(0);
	let loss = $state(NaN);
	let stepMs = $state(0);
	let trainCurve = $state<Array<[number, number]>>([]);
	let valCurve = $state<Array<[number, number]>>([]);

	// NOT $state — a Worker handle must not be proxied.
	let engine: Engine | null = null;
	let playing = false;
	let gen = 0;

	const uniform = $derived(Math.log(config.vocab));
	const canPlay = $derived(phase === 'ready' || phase === 'training');

	async function boot() {
		if (phase !== 'idle') return;
		const myGen = ++gen;
		phase = 'loading';
		loadNote = 'checking for a GPU…';
		try {
			if (!(await detectWebGPU())) {
				phase = 'no-webgpu';
				return;
			}
			if (myGen !== gen) return;
			loadNote = 'building the model on your GPU…';
			const e = new Engine({ tokenData, seed: 42 });
			engine = e;
			await e.init(config);
			if (myGen !== gen) {
				engine = null;
				void e.dispose();
				return;
			}
			valCurve = [[0, await e.valLoss()]];
			phase = 'ready';
		} catch (err) {
			if (myGen !== gen) return;
			errorMsg = err instanceof Error ? err.message : String(err);
			phase = 'error';
		}
	}

	async function toggle() {
		if (phase === 'training') {
			playing = false;
			await engine?.stop();
			if (phase === 'training') phase = 'ready';
			return;
		}
		if (phase !== 'ready' || !engine) return;
		playing = true;
		phase = 'training';
		const myGen = gen;
		while (playing && engine && myGen === gen) {
			const e = engine;
			try {
				await e.train(chunk, (m: TrainMetrics) => {
					if (myGen !== gen) return;
					step = m.step;
					loss = m.loss;
					stepMs = m.stepMs;
					// pushing to a $state array is reactive — no reassignment needed
					trainCurve.push([m.step, m.loss]);
				});
				if (!playing || myGen !== gen) break;
				valCurve.push([step, await e.valLoss()]);
			} catch (err) {
				if (myGen !== gen) return;
				playing = false;
				errorMsg = err instanceof Error ? err.message : String(err);
				phase = 'error';
				return;
			}
		}
	}

	function retry() {
		phase = 'idle';
		errorMsg = '';
		void boot();
	}

	/** Also runs after server prerender — keep it browser-safe. */
	onDestroy(() => {
		gen++;
		playing = false;
		const e = engine;
		engine = null;
		if (e) void e.dispose(); // GPU memory is no souvenir
	});

	/** Build an SVG path from [step, loss] points, log-scaled in y. */
	function path(points: Array<[number, number]>, w = 600, h = 160): string {
		const all = [...trainCurve, ...valCurve].map((p) => p[1]).filter((v) => v > 0);
		if (points.length < 2 || all.length < 2) return '';
		const xMax = Math.max(1, ...[...trainCurve, ...valCurve].map((p) => p[0]));
		const lo = Math.log(Math.min(...all) * 0.9);
		const hi = Math.log(Math.max(...all) * 1.1);
		return points
			.map(([s, v], i) => {
				const x = (s / xMax) * w;
				const y = h - ((Math.log(Math.max(v, 1e-9)) - lo) / (hi - lo)) * h;
				return `${i === 0 ? 'M' : 'L'}${x.toFixed(1)},${y.toFixed(1)}`;
			})
			.join('');
	}
</script>

<!-- the demo boots ~160px before it enters the viewport, exactly once -->
<figure class="plate" use:inview={boot}>
	<header>
		<span class="eyebrow">Plate 1 · training</span>

		{#if canPlay}
			<button class="primary" onclick={toggle}>
				{phase === 'training' ? 'Pause' : 'Train'}
			</button>
		{/if}

		<span class="status num">
			{#if phase === 'loading'}
				{loadNote}
			{:else if phase === 'no-webgpu'}
				needs WebGPU — try Chrome or Edge on desktop
			{:else if phase === 'error'}
				{errorMsg} <button onclick={retry}>Retry</button>
			{:else if step > 0}
				step {step} · loss {loss.toFixed(3)} nats · {stepMs.toFixed(0)} ms/step
			{:else}
				ready
			{/if}
		</span>
	</header>

	<div class="stage">
		{#if phase === 'no-webgpu'}
			<p class="fallback">
				This model trains on your own GPU, and this browser has no WebGPU. Everything below
				describes what it does; the numbers are from a real run.
			</p>
		{:else}
			<svg viewBox="0 0 600 160" preserveAspectRatio="none" role="img"
				aria-label="training loss over steps">
				<path d={path(trainCurve)} class="train" />
				<path d={path(valCurve)} class="val" />
			</svg>
			<p class="caption">
				Training loss in <span class="accent">accent</span>, held-out loss in
				<span class="warm">warm</span>. Knowing nothing would cost {uniform.toFixed(2)} nats
				per token.
			</p>
		{/if}
	</div>
</figure>

<style>
	.plate {
		border: 1px solid var(--line);
		border-radius: 8px;
		background: var(--surface);
		margin: 0;
		overflow: hidden;
	}
	header {
		display: flex;
		align-items: center;
		gap: 0.75rem;
		padding: 0.55rem 0.75rem;
		border-bottom: 1px solid var(--line);
		flex-wrap: wrap;
	}
	.eyebrow {
		text-transform: uppercase;
		letter-spacing: 0.06em;
		font-size: 10px;
		color: var(--ink-3);
	}
	.status {
		margin-left: auto;
		color: var(--ink-2);
		font-size: 11px;
	}
	.num {
		font-variant-numeric: tabular-nums;
	}
	.stage {
		padding: 0.75rem;
	}
	svg {
		width: 100%;
		height: 160px;
		display: block;
	}
	path {
		fill: none;
		vector-effect: non-scaling-stroke;
	}
	.train {
		stroke: var(--accent);
		stroke-width: 1.2;
	}
	.val {
		stroke: var(--warm);
		stroke-width: 1.8;
	}
	.accent {
		color: var(--accent);
	}
	.warm {
		color: var(--warm);
	}
	.caption,
	.fallback {
		color: var(--ink-2);
		font-size: 11px;
		margin: 0.5rem 0 0;
	}
	button {
		font: inherit;
		font-size: 11px;
		padding: 0.2rem 0.65rem;
		border-radius: 5px;
		border: 1px solid var(--line);
		background: transparent;
		color: var(--ink);
		cursor: pointer;
	}
	button.primary {
		background: var(--accent);
		border-color: var(--accent);
		color: var(--paper);
	}
</style>
