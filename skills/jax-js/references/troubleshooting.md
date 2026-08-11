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
0.1.21.

### `TypeError: count.item is not a function`

You put optax inside `jit`. `@jax-js/optax@0.1.2`'s Adam bias correction reads
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

- Chrome/Edge on desktop, Firefox and Safari on macOS 26+, iOS 26+, Android.
- `navigator.gpu` exists but `requestAdapter()` can still return null, and a
  wedged GPU process never answers at all — race it against a timeout and treat
  silence as "no".
- In headless Chromium you need `--enable-unsafe-webgpu --use-angle=metal
  --enable-features=WebGPU`, and the *full* Chromium build rather than the
  headless shell — many CI images have no WebGPU at all. Design the test suite
  so the API contracts run on cpu/wasm in Node and only the integration test
  needs a GPU.

### The boot hangs forever, or the second model never starts

Two workers competing for one `GPUDevice`. A worker that is dropped without
`terminate()` keeps its device, and the next `init()` waits forever.

- **Await** the old engine's `dispose()` before constructing the new one.
- In React StrictMode, guard the effect with a `cancelled` flag — the
  double-mount creates two engines.
- Give every boot step a deadline so this surfaces as an error, not a spinner.

### `postMessage` fails, or the data arrives empty

A transferred `ArrayBuffer` is detached on the sending side. Copy before
transferring anything you still need: `const copy = data.slice()`.

## Behaviour, not errors

### The loss is flat from step 0

In order of likelihood:

1. **Params captured by a jit closure.** `jit((x) => forward(params, x))` bakes
   step-0 weights in as constants. Pass params as an argument.
2. **A zero-initialised output projection** (`wo`, `mlpFc2`, the classifier
   head). Zero there blocks all gradient into the block interior. Use small
   random. See [models.md](models.md#initialisation).
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

Check what the loss is *per what*. A character model at 1.5 nats/token is doing
well; a word-piece model at 1.5 nats/token is barely started. Report the uniform
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

Sampling is running on the training worker. Boot a second worker and courier the
checkpoint. This is the twin-worker pattern
([workers.md](workers.md#the-twin-worker-courier--the-big-win)).

### `stop` does nothing until the run finishes

The training loop is starving the worker's message queue. Yield every few steps:
`if (i % 4 === 3) await new Promise((r) => setTimeout(r, 0));`

### Training is slower than you expected on WebGPU

For small models it genuinely is — dispatch latency dominates and the wasm
backend wins. Measure before assuming. See [performance.md](performance.md).

## When you suspect jax-js itself

1. Reduce to the smallest reproduction — usually under twenty lines.
2. Check it eagerly and under `jit`; the two paths have different coverage.
3. `setDebug(1)` logs kernel compilation; `profiler.startTrace()` gives a kernel
   timeline; `makeJaxpr(f)(x)` prints the traced graph.
4. Compare against the same computation in JAX proper if you can — jax-js
   matches JAX's PRNG bit-for-bit, so seeded tests transfer.
5. Check `FEATURES.md` in the jax-js repo before filing: the gap may be known.
