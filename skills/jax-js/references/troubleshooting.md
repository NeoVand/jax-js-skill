# Troubleshooting

## Error messages

### `ReferenceError: Referenced tracer Array:float32[…] freed, please use .ref move semantics`

A value was consumed twice. **The throw is at the second use; the bug is at the
first.** Work backwards and find what else took ownership. Usual causes, in
order of frequency:

1. `x.item()` (or `.js()`, `.dataSync()`, `await .data()`) followed by
   `x.dispose()`. Reading consumes — drop the dispose.
2. A value used twice in one expression: `np.sum(x.mul(x))` → `np.sum(x.ref.mul(x))`.
3. A pytree handed to both `valueAndGrad` and `solver.update` without
   `tree.ref(params)` on the earlier one.
4. A module-level constant (a mask, a position array) consumed inside a loop.
   `.ref` it per iteration; dispose the master once at the end.
5. Both branches of a conditional read — one branch reads, the other must
   `dispose()`.

Full model in [memory.md](memory.md).

### `Error: jit: routine primitive scatter input is not imm`

`grad` through `np.take` inside `jit`. Use a one-hot matmul for the embedding
(SKILL.md law 4). Eager `grad` through `take` works; jitted does not, as of
0.1.24.

### `TypeError: count.item is not a function`

You put the published Optax Adam update inside `jit`. `@jax-js/optax@0.1.2`'s Adam bias correction reads
its step counter back to the host, which cannot be traced. Keep
`solver.update` / `applyUpdates` outside the jitted function, or use
`templates/fused-adam.ts`.

### `Invalid value "iife" for option "worker.format"` — dev works, `vite build` fails

jax-js lazily imports its wasm/webgpu backends, which is code-splitting, and
Vite's default worker format is `iife`, which cannot split. Add to
`vite.config.ts`:

```ts
worker: { format: 'es' },
optimizeDeps: { include: ['@jax-js/jax', '@jax-js/optax'] },
```

This one only shows up at deploy time, so set it before you write the worker.

### `dot: shapes not aligned along contracting dims: [a] != [b]`

A layer's input width does not match its weight matrix. Print the shapes: they
are free to read (`x.shape` does not consume). Common cause: forgetting that a
VAE waist emits `2 × latent` while the next layer expects `latent`.

### `WebGPU unavailable` / `requestAdapter()` returns null

- Test on HTTPS or localhost. `navigator.gpu` and `requestAdapter()` availability
  depend on browser, operating system and drivers; feature-detect instead of
  maintaining a browser-name allowlist.
- The adapter can be null or initialization can fail. Bound boot attempts with
  a deadline and show the selected fallback or a recoverable error.
- Headless browser support differs from an interactive browser. Check the adapter
  **after navigating to localhost**, not on `about:blank`. Report GPU tests as
  skipped when unavailable; a CPU/wasm pass is not a WebGPU pass.

### Boot or reset hangs, or an old result appears after reset

Inspect ownership and operation ordering rather than assuming device contention:

- Dispose old engines and terminate workers; leaked workers retain resources.
- Serialize worker model operations across `await`, with stop outside the queue.
- Reject pending RPCs on shutdown or transport errors; clearing the map alone
  leaves promises unresolved.
- Guard each late result by generation/checkpoint revision. Cleanup must reach
  an engine that is still initializing.
- A timeout does not cancel the underlying work: dispose the timed-out engine.

### `postMessage` fails, or the data arrives empty

A transferred `ArrayBuffer` is detached on the sending side. Copy before
transferring anything you still need: `const copy = data.slice()`.

## Behaviour, not errors

### The loss is flat from step 0

Check these causes:

1. **Params captured by a jit closure.** `jit((x) => forward(params, x))` bakes
   step-0 weights in as constants. Pass params as an argument.
2. **Gradient flow and initialization.** A zero output projection blocks the
   preceding branch initially, but can itself learn. Inspect per-layer gradient
   norms; do not infer permanently stalled learning from zero init alone.
3. Learning rate far too small, or Adam's `b2` too high for a short run.
4. The loss does not actually depend on the parameters — check that the
   differentiated argument is the one you think it is (`argnums`).

### The loss goes to NaN

- Learning rate too high; halve it.
- `log` or `div` of something that can reach zero — add an epsilon
  (`np.sqrt(ms.add(1e-5))`, not `np.sqrt(ms)`).
- Missing normalisation: an unnormalised residual stream diverges within a few
  hundred steps at these widths.
- Softmax over unbounded logits — use `nn.logSoftmax`, which is stabilised, not
  `np.log(nn.softmax(...))`.
- In RL: a per-token weight that was not normalised by the number of credited
  tokens.

### The loss falls and then the samples are still gibberish

Check what the loss is *per what*. Its interpretation depends on tokenization, corpus and baseline; the same
nats/token value is not comparable across different tokenizers. Report the uniform
baseline (`Math.log(vocab)`) next to it. And confirm the sampler reads the same
weights the trainer wrote — a stale checkpoint in a sampling worker looks exactly
like a model that will not learn.

### Steps get slower over time, then the tab dies

A leak. Every array created inside the loop must be consumed or disposed inside
it. See [memory.md](memory.md#finding-a-leak). Prime suspects: batch tensors,
one-hots, masks, and the loss on steps where it is not read.

### The first step takes seconds, the rest are fast

That is jit compiling. Expected. If it happens *repeatedly*, your input shapes
are changing — pad to fixed sizes. If it happens every step, you passed a
changing value through `staticArgnums`.

### The UI freezes while training

Training on the main thread. Move it to a worker
([workers.md](workers.md)). If it is already in a worker, the UI is probably
re-rendering on every metrics callback — coalesce to one paint per frame.

### The loss curve stutters whenever a sample is drawn

Sampling may be blocking training on the same worker. Try shorter or less frequent
samples, or measure the optional [twin-worker pattern](workers.md#the-twin-worker-courier--optional-concurrent-inference).

### `stop` does nothing until the run finishes

The training loop is starving the worker's message queue. Yield every few steps:
`if (i % 4 === 3) await new Promise((r) => setTimeout(r, 0));`

### Training is slower than you expected on WebGPU

Dispatch latency can dominate small models, letting wasm win. Compare the
actual workload and completed-work timings. See [performance.md](performance.md).

## When you suspect jax-js itself

1. Reduce to the smallest reproduction — usually under twenty lines.
2. Check it eagerly and under `jit`; the two paths have different coverage.
3. `setDebug(1)` logs kernel compilation; `profiler.startTrace()` gives a kernel
   timeline; `makeJaxpr(f)(x)` prints the traced graph.
4. Compare against the same computation in JAX proper if you can — jax-js
   matches JAX's PRNG bit-for-bit, so seeded tests transfer.
5. Check `FEATURES.md` in the jax-js repo before filing: the gap may be known.
