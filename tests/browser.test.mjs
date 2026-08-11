// Integration test: serve examples/ with vite, drive both pages in Chromium,
// and assert the models actually train.
//
//   node tests/browser.test.mjs
//
// WebGPU in headless Chromium is fragile and absent from many CI images, so the
// GPU-only page reports SKIP rather than failing when navigator.gpu is missing.
// The API contracts that matter are covered by tests/api.test.mjs, which runs
// anywhere.

import { spawn } from 'node:child_process';
import { chromium } from 'playwright';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const PORT = 5187;
const args = ['--enable-unsafe-webgpu', '--use-angle=metal', '--enable-features=WebGPU'];

const server = spawn(
	'npm',
	['--prefix', 'examples', 'run', 'dev', '--', '--port', String(PORT), '--strictPort'],
	{ cwd: root, stdio: 'ignore' }
);

const wait = (ms) => new Promise((r) => setTimeout(r, ms));
let failures = 0;
const report = (state, name, detail = '') => {
	if (state === 'FAIL') failures++;
	console.log(`${state.padEnd(4)} ${name}${detail ? `\n       ${detail}` : ''}`);
};

let browser;
try {
	// give vite time to boot
	for (let i = 0; i < 30; i++) {
		try {
			const r = await fetch(`http://localhost:${PORT}/`);
			if (r.ok) break;
		} catch {
			/* not up yet */
		}
		await wait(500);
	}

	browser = await chromium.launch({ headless: true, args });
	const page = await browser.newPage();
	const pageErrors = [];
	page.on('pageerror', (e) => pageErrors.push(String(e)));

	const hasGpu = await page.evaluate(async () => {
		if (typeof navigator === 'undefined' || !navigator.gpu) return false;
		try {
			return (await navigator.gpu.requestAdapter()) !== null;
		} catch {
			return false;
		}
	});
	console.log(`WebGPU in this browser: ${hasGpu ? 'yes' : 'no'}\n`);

	// ── 1. the standalone MLP (runs on wasm too) ──────────────────────────────
	await page.goto(`http://localhost:${PORT}/sine.html`, { waitUntil: 'load' });
	let text = '';
	const deadline = Date.now() + 90_000;
	while (Date.now() < deadline) {
		// The main thread is doing synchronous dispatches; innerText can time out.
		try {
			text = await page.locator('#out').innerText({ timeout: 20_000 });
		} catch {
			continue;
		}
		if (/error:/i.test(text)) break;
		const m = [...text.matchAll(/step\s+(\d+)\s+loss\s+([\d.]+)/g)];
		if (m.length >= 2) break;
		await wait(1500);
	}
	const steps = [...text.matchAll(/step\s+(\d+)\s+loss\s+([\d.]+)/g)];
	if (/error:/i.test(text) || pageErrors.length) {
		report('FAIL', 'sine.html trains', `${text.split('\n').slice(0, 3).join(' | ')} ${pageErrors[0] ?? ''}`);
	} else if (steps.length < 2) {
		report('FAIL', 'sine.html trains', `no progress: ${text.slice(0, 120)}`);
	} else {
		const first = Number(steps[0][2]);
		const last = Number(steps.at(-1)[2]);
		if (!(last < first))
			report('FAIL', 'sine.html loss decreases', `${first} → ${last}`);
		else report('PASS', 'sine.html trains and the loss decreases', `${first} → ${last}`);
	}

	// ── 2. the twin-worker transformer (WebGPU only) ──────────────────────────
	if (!hasGpu) {
		report('SKIP', 'gpt.html twin-worker training', 'no WebGPU in this browser build');
	} else {
		pageErrors.length = 0;
		await page.goto(`http://localhost:${PORT}/gpt.html`, { waitUntil: 'load' });
		let snap = null;
		const gptDeadline = Date.now() + 120_000;
		while (Date.now() < gptDeadline) {
			await wait(2000);
			snap = await page.evaluate(() => {
				const l = window.lab;
				if (!l) return null;
				return {
					phase: l.phase,
					step: l.step,
					loss: l.lossNow,
					samples: l.samples.length,
					err: l.errorMsg,
					firstLoss: l.trainLoss[0]?.[1] ?? NaN,
					valPoints: l.valPoints.length
				};
			});
			if (!snap) continue;
			if (snap.phase === 'error' || snap.phase === 'no-webgpu') break;
			// enough progress to judge: several bursts and at least two samples,
			// which can only both happen if sampling did not stall training
			if (snap.step >= 200 && snap.samples >= 2) break;
		}

		if (!snap || snap.phase === 'error') {
			report('FAIL', 'gpt.html twin-worker training', snap?.err ?? 'never initialised');
		} else if (snap.step < 200) {
			report('FAIL', 'gpt.html reaches 200 steps', JSON.stringify(snap));
		} else if (!(snap.loss < snap.firstLoss)) {
			report('FAIL', 'gpt.html loss decreases', `${snap.firstLoss} → ${snap.loss}`);
		} else if (snap.samples < 2) {
			report('FAIL', 'gpt.html samples while training', `only ${snap.samples} samples`);
		} else {
			report(
				'PASS',
				'gpt.html trains AND samples concurrently',
				`step ${snap.step} · loss ${snap.firstLoss.toFixed(2)} → ${snap.loss.toFixed(2)} nats · ` +
					`${snap.samples} samples · ${snap.valPoints} eval points`
			);
		}
		if (pageErrors.length) report('FAIL', 'gpt.html has no page errors', pageErrors[0]);
		else report('PASS', 'gpt.html has no page errors');
	}
} finally {
	await browser?.close();
	server.kill('SIGTERM');
}

console.log('');
console.log(failures === 0 ? 'browser tests: ok' : `browser tests: ${failures} failure(s)`);
process.exit(failures ? 1 : 0);
