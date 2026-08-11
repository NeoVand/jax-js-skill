// Example 2 — the twin-worker lab, wired to a page.
//
// This is the architecture that matters: training in one worker, sampling in a
// second, and a UI that never waits for either. Watch the loss curve while a
// sample is being written — it does not pause.

import { TwinLab } from '../../skills/jax-js/templates/twin-engine';
import { encodePrompt } from '../../skills/jax-js/templates/tokens';
import { buildCorpus } from './corpus';
import { drawLossChart } from './chart';

const go = document.getElementById('go') as HTMLButtonElement;
const resetBtn = document.getElementById('reset') as HTMLButtonElement;
const status = document.getElementById('status') as HTMLSpanElement;
const note = document.getElementById('sampler-note') as HTMLSpanElement;
const chart = document.getElementById('chart') as HTMLCanvasElement;
const samplesEl = document.getElementById('samples') as HTMLDivElement;
const out = document.getElementById('out') as HTMLPreElement;

const corpus = buildCorpus(120_000, 7);
const CFG = { nLayer: 2, nEmbd: 96, nHead: 4, blockSize: 96, vocab: corpus.chars.length };

// Text becomes token IDs here, in the application layer, and is validated on the
// way. Everything downstream — the lab, the engine, the worker — speaks integers
// only, so there is no route from free text into model execution.
const BOUNDS = { vocab: CFG.vocab, maxLen: Math.floor(CFG.blockSize / 2) };
const AUTO_PROMPT_TOKENS = encodePrompt('the ', corpus.encode, BOUNDS);

let raf = 0;
const lab = new TwinLab({
	config: CFG,
	tokenData: corpus.tokens,
	decode: corpus.decode,
	lr: 1.5e-3,
	chunk: 40,
	autoPromptTokens: AUTO_PROMPT_TOKENS,
	notify: () => {
		// Coalesce state changes into one paint per frame — a metrics callback
		// fires every step and rendering on each would be the new bottleneck.
		if (raf) return;
		raf = requestAnimationFrame(() => {
			raf = 0;
			render();
		});
	}
});

function render() {
	const uniform = Math.log(CFG.vocab);
	status.textContent =
		lab.phase === 'loading'
			? lab.loadNote || 'loading…'
			: lab.phase === 'error'
				? lab.errorMsg
				: lab.phase === 'no-webgpu'
					? 'no WebGPU in this browser'
					: `step ${lab.step} · loss ${Number.isFinite(lab.lossNow) ? lab.lossNow.toFixed(3) : '—'} nats` +
						` · ${Math.round(lab.tokensPerSec).toLocaleString()} tok/s`;

	go.disabled = lab.phase !== 'ready' && lab.phase !== 'training';
	resetBtn.disabled = lab.phase !== 'ready' && lab.phase !== 'training';
	go.textContent = lab.phase === 'training' ? 'Pause' : 'Train';

	drawLossChart(chart, lab.trainLoss, lab.valPoints);

	samplesEl.replaceChildren(
		...lab.samples.map((s) => {
			const div = document.createElement('div');
			div.className = 'sample';
			const step = document.createElement('div');
			step.className = 'step';
			step.textContent = `step ${s.step}`;
			const body = document.createElement('div');
			body.textContent = s.prompt + s.text;
			div.append(step, body);
			return div;
		})
	);

	out.textContent =
		`corpus ${corpus.tokens.length.toLocaleString()} chars · vocab ${CFG.vocab} · ` +
		`${lab.paramCount.toLocaleString()} params · uniform guess = ${uniform.toFixed(2)} nats\n` +
		(lab.phase === 'error' ? `error: ${lab.errorMsg}` : '');

	// e2e markers
	document.body.dataset.phase = lab.phase;
	document.body.dataset.step = String(lab.step);
	document.body.dataset.loss = String(lab.lossNow);
	document.body.dataset.samples = String(lab.samples.length);
}

// Exposed for the console and for e2e: rAF is throttled to zero in a hidden
// tab, so the DOM is not a reliable place to read progress from.
(window as unknown as { lab: TwinLab }).lab = lab;

go.addEventListener('click', () => lab.toggle());
resetBtn.addEventListener('click', () => void lab.reset());
window.addEventListener('pagehide', () => lab.disposeAll());

void (async () => {
	await lab.probe();
	render();
	await lab.boot();
	render();
	if (lab.phase === 'ready') lab.start(); // auto-start so the page proves itself
})();
