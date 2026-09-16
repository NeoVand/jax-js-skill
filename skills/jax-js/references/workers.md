# Workers, RPC and model lifecycle

Put sustained training in a worker so compilation, dispatch and readbacks do not
block page input and paint. Initialize JaxJS inside each worker. Each owns its
own device, arrays, optimizer, compiled functions and data.

```
UI component → Engine (promise RPC) → Worker (model owner)
```

Keep arrays inside the worker. Send plain metrics and transferable buffers.
Match responses to numeric request IDs; metrics events stream without settling
the request. `templates/engine.ts` implements this contract.

## Serialize access to the model

An `async onmessage` is not a mutex: another message can run while training
awaits a yield or readback. Serialize model operations. Let `stop` bypass the
queue and set a flag; otherwise it waits behind the loop it is meant to stop.

```ts
let queue = Promise.resolve();
self.onmessage = (e: MessageEvent<RpcRequest>) => {
  const req = e.data;
  if (req.op === 'stop') {
    stopRequested = true;
    post({ id: req.id, ok: true, result: {} });
    return;
  }
  if (req.op === 'dispose') stopRequested = true;
  queue = queue.then(() => dispatch(req)); // dispatch catches and replies to errors
};
```

`train`, `load`, `init`, evaluation, export and disposal must not mutate or read
partially replaced state. The training handler yields periodically and checks
`stopRequested`. A stop acknowledgement means the flag was set; await the active
training promise or a subsequent queued operation to know training has ended.
Avoid unbounded queues: the UI should allow one training request at a time.

A handler can return `{ checkpoint, __transfer: [checkpoint] }`. The dispatcher
removes `__transfer` and posts the result with the transfer list. A thrown handler
error must reply to that request and leave the queue usable.

## Validate the data contract

Check configuration dimensions, finite numeric values, dataset lengths, token
ranges and checkpoint shapes before using them to allocate arrays. Adapt those
checks to the model; TypeScript casts do not validate runtime messages.

The language-model templates accept integer token IDs. `toPromptTokens()` checks
integrality, vocabulary bounds and context length, catching tokenizer/model
mismatches. This is a template contract, not a ban on worker-side tokenization.
Tokenizing text does not neutralize its meaning or prevent prompt injection.
Always display generated text through escaping, as described in [ui.md](ui.md).

## Transferables

Transferring an `ArrayBuffer` detaches it on the sender. Copy data that remains
needed there:

```ts
const copy = tokenData.slice();
await this.call('init', { tokenData: copy.buffer }, [copy.buffer]);
```

Transfer large datasets, checkpoints and activation dumps. This avoids a buffer
clone, but GPU readback and upload are still real costs. Handle synchronous
`postMessage` exceptions by removing and rejecting that pending request.

## Disposal

Use the tested implementation in `templates/engine.ts`:

1. Send a graceful disposal request, then stop accepting new calls.
2. Reject pending work so callers cannot hang; stale-result guards suppress
   expected cancellation errors in unmounted or reset views.
3. Give graceful cleanup a bounded deadline, then terminate in `finally`.
4. Clear the deadline and reject any remaining requests before clearing the map.
5. Make repeated `dispose()` calls share one shutdown operation.

Handle worker `error` and `messageerror` the same way: reject pending work and
close the unusable worker. Await disposal before replacing an engine when
practical. Leaked workers consume resources; multiple workers do **not**
inherently deadlock on one shared GPUDevice.

## Generation counters and deadlines

Tag each boot, reset and evaluation with a generation/revision. Check it after
**every** await before writing UI state, including validation and sampling.
Do the same in streaming metrics callbacks.

```ts
const myGen = ++generation;
const engine = new Engine(options);
currentEngine = engine; // cleanup can reach it even during init
try {
  await guard('model initialization', engine.init(config));
  if (myGen !== generation) return;
  const value = await engine.valLoss();
  if (myGen !== generation) return;
  publish(value);
} finally {
  if (myGen !== generation) await engine.dispose();
}
```

Increment generation before resetting or unmounting. A timeout only stops
waiting; it does not cancel the underlying initialization. Dispose failed or
superseded engines too, and offer Retry for recoverable boot failures.

```ts
async function guard<T>(what: string, promise: Promise<T>, ms = 25_000): Promise<T> {
  let timer: ReturnType<typeof setTimeout> | undefined;
  try {
    return await Promise.race([promise, new Promise<never>((_, reject) => {
      timer = setTimeout(() => reject(new Error(`${what} timed out`)), ms);
    })]);
  } finally { clearTimeout(timer); }
}
```

## The twin-worker courier — optional concurrent inference

For expensive samples while training continues, `templates/twin-engine.ts`
uses a trainer and sampler. At a burst boundary, export weights and load them
into the sampler; let the next burst overlap with inference. Keep the sampler's
load/sample operations serialized and allow only one sample job at a time.

Record the checkpoint step at export, rather than labeling the result with the
trainer's later step when sampling finishes. Drop results after reset. Fall back
to inline inference if a second worker cannot initialize.

Budget the cost: 5M float32 weights alone occupy about 20 MB per copy, plus
activations, compilation caches, temporary buffers, and optimizer state if the
sampler initializes it. Separate devices still contend for physical GPU
resources; overlap does not guarantee better throughput. A single worker with
short pauses between bursts is often enough.

## Bundlers

```ts
new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' });
```

Vite/SvelteKit resolve this relative module URL. A plain string path is not
bundled the same way. Set `worker: { format: 'es' }` and test production output,
not only the dev server.
