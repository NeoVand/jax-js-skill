# Workers: the trainer, the RPC, and the sampler that keeps training alive

## Why a worker

Training on the main thread janks the page even on WebGPU, because every loss
readback blocks the thread that also runs layout and paint. Moving the model into
a Web Worker measured *faster* than the main thread in practice, with the UI
holding 60fps.

Devices do not cross threads: `init()` and `defaultDevice()` are called **inside**
the worker, and each worker gets its own `GPUDevice`.

## The three layers

```
UI component  ──▶  Engine (main thread, promise RPC)  ──▶  Worker (owns jax-js)
```

The engine is a plain class, not framework state. The worker owns params,
optimizer state, jitted functions and the dataset. The UI never sees an array.

## The RPC protocol

One numeric id per call; a `metrics` event streams progress without resolving.

**Worker side** — `templates/worker.ts`:

```ts
interface RpcRequest { id: number; op: string; [k: string]: unknown }

const post = (msg: unknown, transfer?: Transferable[]) =>
  (self as unknown as Worker).postMessage(msg, { transfer: transfer ?? [] });

const handlers: Record<string, (req: RpcRequest) => unknown | Promise<unknown>> = {
  init: handleInit, train: handleTrain, stop: () => { stopRequested = true; return {}; },
  predict: handlePredict, export: handleExport, load: handleLoad, dispose: handleDispose,
};

self.onmessage = async (e: MessageEvent<RpcRequest>) => {
  const req = e.data;
  try {
    const handler = handlers[req.op];
    if (!handler) throw new Error(`unknown op: ${req.op}`);
    const result = (await handler(req)) as Record<string, unknown> & { __transfer?: Transferable[] };
    const transfer = result?.__transfer;
    if (transfer) delete result.__transfer;
    post({ id: req.id, ok: true, result }, transfer);
  } catch (err) {
    post({ id: req.id, ok: false, error: err instanceof Error ? `${err.name}: ${err.message}` : String(err) });
  }
};
```

The `__transfer` convention lets a handler return `{ buf, __transfer: [buf] }` and
have the dispatcher move the buffer instead of copying it.

**Main-thread side** — `templates/engine.ts`:

```ts
private call<T>(op: string, payload = {}, transfer: Transferable[] = [],
                onMetrics?: (m: Metrics) => void): Promise<T> {
  const id = this.nextId++;
  return new Promise<T>((resolve, reject) => {
    this.pending.set(id, { resolve, reject, onMetrics });
    this.worker.postMessage({ id, op, ...payload }, transfer);
  });
}

private onMessage(e: MessageEvent) {
  const msg = e.data;
  const p = this.pending.get(msg.id);
  if (!p) return;
  if (msg.event === 'metrics') { p.onMetrics?.(msg.m); return; }   // stream, don't settle
  this.pending.delete(msg.id);
  msg.ok ? p.resolve(msg.result) : p.reject(new Error(msg.error));
}
```

### The worker validates everything it receives

The main thread is not a trust boundary. Every RPC field gets checked on
receipt: token sequences through `toPromptTokens()` (integers inside the
vocabulary, length-capped), numeric knobs clamped to sane ranges.

```ts
const prompt = toPromptTokens(req.promptTokens ?? [], {
  vocab: c.vocab, maxLen: Math.floor(c.blockSize / 2) });
const temperature = num(req.temperature, 0.8, 1e-4, 100);
const topK = Math.floor(num(req.topK, 40, 0, c.vocab));
```

The message contract is **integer token IDs, never text** — encoding happens in
the application layer (`templates/tokens.ts`). That keeps arbitrary strings out
of model execution entirely, and turns an encoder/vocabulary mismatch into a
thrown error instead of silently corrupt one-hots.

### Transferables

A transferred `ArrayBuffer` is **detached** on the sending side. If the caller
still needs the data, copy first:

```ts
const copy = tokenData.slice();                     // detach the copy, not the original
await this.call('init', { tokenData: copy.buffer }, [copy.buffer]);
```

Transfer everything big: datasets in, checkpoints and activation dumps out.

### Making `stop` land

A tight training loop starves the worker's own message queue, so the `stop`
message never gets processed. Yield every few steps:

```ts
for (let i = 0; i < steps; i++) {
  if (stopRequested) break;
  const t0 = performance.now();
  const loss = trainStep();
  post({ id: req.id, event: 'metrics', m: { step: ++stepCounter, loss, stepMs: performance.now() - t0 } });
  if (i % 4 === 3) await new Promise((r) => setTimeout(r, 0));   // let 'stop' through
}
```

### Disposal

A worker mid-`jit` answers no RPC promptly, and a worker that never releases its
`GPUDevice` blocks the next one from acquiring it. Give the graceful path a
deadline and terminate either way:

```ts
async dispose(): Promise<void> {
  try {
    await Promise.race([this.call('dispose'), new Promise((r) => setTimeout(r, 400))]);
  } finally {
    this.worker.terminate();
    this.pending.clear();
  }
}
```

When rebuilding a model, **await** the old engine's `dispose()` before
constructing the new one. Two workers racing for one GPU device deadlocks the
page.

---

## The twin-worker courier — the big win

**Problem.** The UI wants to show what the model can do *while it trains*: write a
text sample, evaluate a position, render an attention map. If those calls go to
the training worker, training stops for the duration. On a transformer that is
hundreds of milliseconds per sample, and the loss curve visibly stutters.

**Solution.** Boot a **second** worker with the same config. After each training
burst, export a checkpoint from the trainer and load it into the sampler. The
sampler answers the UI's inference calls on its own GPU device while the trainer
keeps stepping.

```ts
class Lab {
  private engine: Engine | null = null;     // trains
  private sampler: Engine | null = null;    // answers, never trains
  private samplerReady = false;

  /** Which engine writes the next sample: the sampler, freshly loaded with the
   *  trainer's current weights (training keeps running), or the trainer itself
   *  when no sampler exists (the caller then waits, as it always used to). */
  private async sampleEngine(): Promise<Engine | null> {
    const e = this.engine;
    const s = this.samplerReady ? this.sampler : null;
    if (!e || !s) return e;
    try {
      const ckpt = await e.exportCheckpoint();   // one quick readback
      await s.loadWeights(ckpt);                 // transferred, not copied
      return s;
    } catch {
      this.samplerReady = false; this.sampler = null; void s?.dispose();
      return this.engine;                        // degrade to inline sampling
    }
  }

  private async loop() {
    while (this.playing) {
      await this.engine!.train(CHUNK, (m) => this.record(m));
      const v = await this.engine!.valLoss();
      this.valPoints.push([this.step, v]);
      // With the sampler up, the sample is written on its own device while the
      // next burst runs — the curve never pauses. Without it, wait inline.
      if (this.samplerReady) void this.autoSample();
      else await this.autoSample();
    }
  }
}
```

Four properties make this work:

- **Boot the sampler in the background, silently.** If it fails, sampling falls
  back to the training worker — slower, never broken.
- **Courier after the burst, before the next `train()` call.** Then the sample is
  exactly the weights of the step it is labelled with. Honest, not approximate.
- **`loadWeights` must be an in-place op**, not a re-init: it keeps the config,
  the jit caches and the dataset, so it costs one buffer transfer. A full re-init
  re-uploads the corpus.
- **Never block the loop on the sampler.** `void this.autoSample()` — fire and
  forget, guarded by a generation counter (below).

### Cost

One extra GPU device and one extra copy of the weights. For a 5M-parameter model
that is ~20 MB — cheap for a loss curve that never stutters.

### When *not* to use it

If the UI only inspects the model while training is paused (a scrubber, a
post-hoc probe), one worker is simpler and correct. Reach for the twin when
inference and training genuinely overlap in time.

---

## Generation counters

Anything async can land after the model it belonged to has been replaced or
disposed. A monotonically increasing `gen` makes every late arrival a no-op.

```ts
private gen = 0;

async boot() {
  const myGen = ++this.gen;
  ...
  const result = await something();
  if (myGen !== this.gen) return;      // superseded — drop it on the floor
  this.state = result;
}

disposeAll() {
  this.gen++;                          // cancels every in-flight callback
  void this.engine?.dispose();
  this.engine = null;
}
```

Also guard the *worker handle itself*, so a superseded boot hands its device
back:

```ts
const superseded = () => {
  if (myGen === this.gen) return false;
  if (this.engine === engine) this.engine = null;
  void engine.dispose();               // give the GPU device back
  return true;
};
await engine.init(config);
if (superseded()) return;
```

## Deadlines on every boot step

A stalled fetch or a GPU device that never arrives leaves the UI saying
"loading…" forever. Give each step a deadline so the user gets an error and a
Retry button:

```ts
function guard<T>(what: string, p: Promise<T>, ms = 25_000): Promise<T> {
  return Promise.race([
    p,
    new Promise<never>((_, reject) =>
      setTimeout(() => reject(new Error(`${what} timed out — try again`)), ms)),
  ]);
}

const tok = await guard('the corpus', loadCorpus());
await guard('the GPU', engine.init(config));
```

## Bundlers

Vite and SvelteKit resolve module workers from a URL relative to the importing
file:

```ts
this.worker = new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' });
```

This works in dev and in build. Do not use a string path; it will not be bundled.
