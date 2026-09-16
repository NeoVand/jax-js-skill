// Lifecycle contracts: promises must settle on shutdown and transport failure.
// Exercise both the shipped transformer engine and the generated MLP starter.
import { describe, test, after } from 'node:test';
import assert from 'node:assert/strict';
import { readFile, mkdtemp, rm } from 'node:fs/promises';
import { execFileSync } from 'node:child_process';
import { stripTypeScriptTypes } from 'node:module';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const root = fileURLToPath(new URL('../', import.meta.url));
const scratch = await mkdtemp(join(tmpdir(), 'jax-skill-rpc-'));
const starter = join(scratch, 'starter');
execFileSync(process.execPath, [join(root, 'skills/jax-js/scripts/scaffold.mjs'), starter]);
after(() => rm(scratch, { recursive: true, force: true }));

async function loadEngine(path) {
	const url = pathToFileURL(path).href;
	const source = (await readFile(path, 'utf8'))
		.replaceAll('import.meta.url', JSON.stringify(url))
		.replace("'./tokens'", JSON.stringify(new URL('./tokens.ts', url).href));
	const js = stripTypeScriptTypes(source);
	return (await import(`data:text/javascript;base64,${Buffer.from(js).toString('base64')}`)).Engine;
}

class FakeWorker {
	static latest;
	sent = [];
	terminated = 0;
	failSend = false;
	onmessage = null;
	onerror = null;
	onmessageerror = null;
	constructor() { FakeWorker.latest = this; }
	postMessage(message) {
		if (this.failSend) throw new Error('structured clone failed');
		this.sent.push(message);
	}
	terminate() { this.terminated++; }
	reply(message, result = {}) {
		this.onmessage?.({ data: { id: message.id, ok: true, result } });
	}
}

const originalWorker = globalThis.Worker;
globalThis.Worker = FakeWorker;
after(() => { globalThis.Worker = originalWorker; });

const TemplateEngine = await loadEngine(join(root, 'skills/jax-js/templates/engine.ts'));
const StarterEngine = await loadEngine(join(starter, 'src/engine.ts'));
for (const [name, create] of [
	['template', () => new TemplateEngine({ tokenData: new Uint16Array([0, 1]) })],
	['scaffold', () => new StarterEngine()]
]) {
	describe(`${name} RPC lifecycle`, () => {
		test('metrics stream before the training promise settles', async () => {
			const engine = create(), worker = FakeWorker.latest, metrics = [];
			let settled = false;
			const training = engine.train(3, (m) => metrics.push(m)).then(() => { settled = true; });
			const request = worker.sent[0];
			worker.onmessage({ data: { id: request.id, event: 'metrics', m: { step: 1, loss: 2 } } });
			await Promise.resolve();
			assert.equal(settled, false);
			assert.equal(metrics[0].step, 1);
			worker.reply(request);
			await training;
			const disposing = engine.dispose();
			worker.reply(worker.sent.at(-1));
			await disposing;
		});

		test('dispose rejects pending calls, blocks new work and is idempotent', async () => {
			const engine = create(), worker = FakeWorker.latest;
			const training = assert.rejects(engine.train(99, () => {}), /disposed/);
			const disposing = engine.dispose();
			assert.equal(engine.dispose(), disposing);
			await training;
			await assert.rejects(engine.stop(), /disposed|closed/);
			assert.equal(worker.sent.filter((m) => m.op === 'dispose').length, 1);
			worker.reply(worker.sent.at(-1));
			await disposing;
			assert.equal(worker.terminated, 1);
			assert.equal(engine.pending.size, 0);
		});

		test('a worker that never replies is terminated after the grace period', async () => {
			const engine = create(), worker = FakeWorker.latest;
			const stopped = assert.rejects(engine.stop(), /disposed/);
			await engine.dispose();
			await stopped;
			assert.equal(worker.terminated, 1);
			assert.equal(engine.pending.size, 0);
		});

		for (const event of ['onerror', 'onmessageerror']) {
			test(`${event} rejects callers and closes the worker`, async () => {
				const engine = create(), worker = FakeWorker.latest;
				const stopped = assert.rejects(engine.stop(), /worker/);
				worker[event]({ message: 'worker crashed' });
				await stopped;
				await assert.rejects(engine.stop(), /closed/);
				await engine.dispose();
				assert.equal(worker.terminated, 1);
				assert.equal(engine.pending.size, 0);
			});
		}

		test('synchronous postMessage failure removes its pending request', async () => {
			const engine = create(), worker = FakeWorker.latest;
			worker.failSend = true;
			await assert.rejects(engine.stop(), /structured clone/);
			assert.equal(engine.pending.size, 0);
			worker.failSend = false;
			const disposing = engine.dispose();
			worker.reply(worker.sent.at(-1));
			await disposing;
		});
	});
}
