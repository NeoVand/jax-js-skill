---
name: jax-js
description: Build and train real neural networks in the browser with jax-js (@jax-js/jax) on WebGPU — autodiff, jit, optax optimizers, Web Worker training loops, and interfaces that stay at 60fps while training. Use for any task involving jax-js, @jax-js/jax, @jax-js/optax, "JAX in the browser", client-side or in-browser model training, WebGPU machine learning, or interactive ML demos (MLP, CNN, transformer, autoencoder, VAE, policy gradient, GRPO/RLVR) that run entirely on the user's own GPU.
license: MIT
---

# jax-js — real training, in the browser

jax-js gives you JAX's semantics in TypeScript — `numpy` arrays, `grad`, `jit`,
`vmap`, pytrees, `optax` optimizers — compiled to WebGPU, WebAssembly, WebGL or
plain CPU. No Python, no server, no upload. A 5M-parameter transformer trains at
interactive speed on a laptop GPU, and the weights never leave the page.

It is not TensorFlow.js. There is no `Model.fit()`, no layer objects, no
`tensor.clone()`. Parameters are plain JS objects of arrays; a training step is a
function you differentiate. **And arrays are moved, not shared** — this is the
one thing that breaks every newcomer, including agents. Read the five laws before
writing a line.

---

## Setup

```bash
npm i @jax-js/jax @jax-js/optax
```

```ts
import { init, defaultDevice, numpy as np, nn, jit, valueAndGrad, tree, random } from '@jax-js/jax';
import { adam, applyUpdates } from '@jax-js/optax';

const devices = await init();                 // ['cpu','wasm','webgl','webgpu'] — what this browser has
defaultDevice(devices.includes('webgpu') ? 'webgpu' : 'wasm');
```

Choose `webgpu → wasm` and skip the rest: `cpu` is 100–200× slower than wasm and
`webgl` is a distant third. Below ~100k parameters wasm actually **ties or beats
WebGPU** (dispatch latency dominates), so a small demo needs no GPU at all;
above ~250k WebGPU pulls away fast — 10× at 5M parameters. Measured numbers in
[references/performance.md](references/performance.md).

`init()` must finish before any array is created. In a Web Worker, `init()` and
`defaultDevice()` are called inside the worker — devices do not cross threads.

**Configure Vite before you write the worker.** jax-js lazily imports its
wasm/webgpu backends, so a worker that uses it needs code-splitting. Vite's
default worker format cannot do that: the dev server works and `vite build`
fails with `Invalid value "iife" for option "worker.format"`.

```ts
// vite.config.ts
export default defineConfig({
  worker: { format: 'es' },
  optimizeDeps: { include: ['@jax-js/jax', '@jax-js/optax'] },
});
```

Feature-detect before promising a GPU (`navigator.gpu` exists but
`requestAdapter()` can still return null, and a wedged GPU process never answers
at all):

```ts
export async function detectWebGPU(): Promise<boolean> {
  if (typeof navigator === 'undefined' || !navigator.gpu) return false;
  try {
    const adapter = await Promise.race([
      navigator.gpu.requestAdapter(),
      new Promise<null>((r) => setTimeout(() => r(null), 8000)),   // silence = no
    ]);
    return adapter !== null;
  } catch { return false; }
}
```

---

## The five laws

### 1. Every array is consumed exactly once. Use `.ref` to lend a second use.

Passing an array to any operation **moves** it. Touching it again throws
`Referenced tracer ... freed, please use .ref move semantics`.

```ts
const a = np.array([1, 2, 3]);
np.sum(a);            // a is now GONE
np.sum(a);            // ✗ ReferenceError

const b = np.array([1, 2, 3]);
np.sum(b.ref).item(); // ✓ lends one use
np.sum(b).item();     // ✓ final use consumes it
```

This applies to *everything* that takes an array: `np.*` functions, method calls
(`x.add(y)` consumes both `x` and `y`), `nn.*`, `np.zerosLike(a)`, jitted
functions, `solver.update(...)`, `applyUpdates(...)`.

Counting rule: **count the uses of a value in a scope; add `.ref` to all but the
last.** For pytrees use `tree.ref(params)` and free with `tree.dispose(params)`.

```ts
// params is used twice → ref the first
const [loss, grads] = jitStep(tree.ref(params), batch);
const [updates, next] = solver.update(grads, optState, tree.ref(params));
params = applyUpdates(params, updates);   // last use, consumes
```

### 2. Reading a value consumes it. Never `dispose()` afterwards.

`.item()`, `.js()`, `.dataSync()` and `await .data()` **all consume**.
`blockUntilReady()` does not.

```ts
const l = lossVal.item();       // lossVal is gone — do NOT call lossVal.dispose()
const keep = lossVal.ref.item(); // read a copy if you still need the array
```

The most common jax-js bug in agent-written code is `x.item(); x.dispose();` —
a double free. If a loss is only *sometimes* read, dispose it in the other branch:

```ts
if (step % 50 === 0) log(lossVal.item());
else lossVal.dispose();
```

### 3. `jit` the step, keep shapes constant, pass params as arguments.

`jit` traces once per distinct input **shape/dtype signature** and caches the
compiled kernel. Two consequences:

- **Pad variable-length inputs to a fixed block size.** A generation loop that
  grows the prompt by one token per step will recompile on every step. Right-pad
  to `blockSize` instead; causal attention makes the padding irrelevant.
- **Never close over params.** A jitted closure bakes them in as trace-time
  constants and your model will never improve — the loss curve goes flat and
  looks like a learning-rate bug.

```ts
// ✗ params baked in at trace time — samples never change
const bad = jit((tok, pos) => forward(params, tok, pos));

// ✓ params flow in as an argument
const good = jit((p, tok, pos) => forward(p, tok, pos));
good(tree.ref(params), tok, pos);
```

`jit()` returns an *owned* function; call `.dispose()` on it when the model is
torn down. `staticArgnums` recompiles for **every distinct value** — never pass a
step counter through it.

### 4. Embeddings are one-hot matmuls, not `np.take`.

Gather has a backward pass in eager mode, but **`grad` through `np.take` fails
inside `jit`** (`routine primitive scatter input is not imm`, verified on
0.1.21). Since everything trainable must be jitted, embeddings are:

```ts
const tokenOH = nn.oneHot(inputIds, vocab);        // [B, S, V]
let x = np.dot(tokenOH.reshape([-1, vocab]), params.wte);   // [B·S, D]
```

Build the one-hot **outside** the jitted function and pass it in. This costs
`B·S·V` floats, which is fine to a vocab of a few thousand — the right ceiling
for an in-browser model anyway. `np.take` is still correct for inference-only
paths and for indexing that is never differentiated.

### 5. Train in a Worker. Sample from a *second* Worker.

Training on the main thread janks the page even on WebGPU, because readbacks
block. Put the model in a Web Worker behind a small RPC — measured *faster* than
the main thread, with the UI at 60fps.

Then: any inference the UI wants **while training runs** (a text sample, a board
evaluation, an attention map) must not stop the training loop. Boot a second
worker, courier a checkpoint to it after each burst, and let it answer queries on
its own GPU device while the trainer keeps stepping. This is the single biggest
perceived-performance win in an interactive ML page — see
[references/workers.md](references/workers.md).

---

## The training step, three ways

Pick by how much you need. All three are verified in the skill repo's
[tests/api.test.mjs](https://github.com/NeoVand/jax-js-skill/blob/main/tests/api.test.mjs).

**A. Default — `jit` the loss+grad, run optax outside.** Correct, fast enough for
almost everything, and the only option if you want optax's schedules or chains.

```ts
const jitStep = jit((p: any, x: any, y: any) =>
  valueAndGrad((pp: any) => lossFn(pp, x, y))(p));

const solver = adam(3e-4, { b1: 0.9, b2: 0.99 });
let optState = solver.init(tree.ref(params));

for (let i = 0; i < steps; i++) {
  const { x, y } = nextBatch();                              // fresh device arrays
  const [lossVal, grads] = jitStep(tree.ref(params), x, y);
  const [updates, nextState] = solver.update(grads, optState, tree.ref(params));
  params = applyUpdates(params, updates);
  optState = nextState;
  loss = lossVal.item();                                     // consumes; no dispose
}
```

> `@jax-js/optax@0.1.2` **cannot be placed inside `jit`** — its Adam bias
> correction calls `count.item()`, a host readback, and jit throws
> `count.item is not a function`. Keep `solver.update`/`applyUpdates` outside.

**B. Fastest — fuse the optimizer into the jitted step by hand.** Measured
**2.9× faster on a 235k-parameter transformer on WebGPU** (26.7 → 9.2 ms/step)
and up to 6× on small MLPs, because optax outside `jit` spends most of the step
on per-tensor kernel-dispatch latency. Use `templates/fused-adam.ts`; it
converges bit-for-bit with optax's Adam. Bias-correction constants go in as
**device scalars**, not `staticArgnums`, so the kernel is traced once. Reach for
this whenever the step time matters — see [references/performance.md](references/performance.md).

**C. Simplest — no optimizer at all.** For teaching demos and 2-parameter
landscapes, `valueAndGrad` plus `x = x.sub(g.mul(lr))` is the whole story and
reads beautifully.

### Pacing

Run in bursts (25–50 steps), then yield. A tight `for` loop starves the worker's
own message queue, so a `stop` message never arrives:

```ts
for (let i = 0; i < steps; i++) {
  if (stopRequested) break;
  trainStep();
  if (i % 4 === 3) await new Promise((r) => setTimeout(r, 0));  // let 'stop' land
}
```

Sync the loss **every step**. Letting steps queue up and blocking once at the end
measures ~2× *slower* on WebGPU, not faster (upstream issue #151, still open).

---

## Choosing a shape for the app

| Situation | Build |
| --- | --- |
| Teaching script, one canvas, ≤100k params | Single module on the main thread; `await new Promise(r => setTimeout(r))` between bursts. `templates/standalone-lab.ts` |
| Any real training (MLP on MNIST and up) | Worker + promise-RPC engine. `templates/worker.ts` + `templates/engine.ts` |
| UI must stay live *during* training (sampling, probing, playing) | Twin workers: trainer + sampler, checkpoint couriered between them. [references/workers.md](references/workers.md) |
| Framework UI (Svelte/React) | Engine in a module-level store; never put the engine in reactive state. [references/ui.md](references/ui.md) |

---

## Reference index

Read the file that matches the task; do not read them all.

| I need to… | Read |
| --- | --- |
| Look up an operator, dtype, random key, or shape rule | [references/api.md](references/api.md) |
| Understand or debug ownership, leaks, `already freed` errors | [references/memory.md](references/memory.md) |
| Structure the worker, the RPC, the twin-worker sampler | [references/workers.md](references/workers.md) |
| Write an MLP, CNN, transformer, autoencoder or VAE | [references/models.md](references/models.md) |
| Do RL: REINFORCE, policy gradient, GRPO/RLVR, DPO | [references/rl.md](references/rl.md) |
| Build the UI: charts, canvases, controls, lifecycle, theming | [references/ui.md](references/ui.md) |
| Make it faster, or find out why it is slow | [references/performance.md](references/performance.md) |
| Fix an error message | [references/troubleshooting.md](references/troubleshooting.md) |

## Templates

Copy, don't retype. Each file is runnable and tested.

| File | What it is |
| --- | --- |
| `templates/standalone-lab.ts` | Whole training loop in one file: pytree params, `valueAndGrad`, Adam, canvas |
| `templates/worker.ts` | Worker that owns the model; RPC dispatch, transferables, `stop` handling |
| `templates/engine.ts` | Main-thread promise-RPC client with streaming metrics |
| `templates/twin-engine.ts` | Trainer + sampler pair; checkpoint courier so training never pauses |
| `templates/tokens.ts` | The token boundary: encode text in the app, hand the worker validated integer IDs |
| `templates/model-mlp.ts` | Configurable MLP: activations, mse/xent, VAE bottleneck |
| `templates/model-transformer.ts` | Decoder-only transformer: init, forward, loss, sampling, attention capture |
| `templates/fused-adam.ts` | Optimizer fused inside `jit` |
| `templates/ui-svelte5.svelte` | Svelte 5 runes plate: lifecycle, phases, loss chart, controls |
| `templates/ui-react.tsx` | Same contract in React |

## Scripts

```bash
node scripts/doctor.mjs            # verify jax-js version + API assumptions in this project
node scripts/scaffold.mjs <dir>    # write a runnable vite + jax-js + worker starter
```

---

## Before you say it works

1. `node scripts/doctor.mjs` — passes on the installed version.
2. Loss **goes down**. A flat curve almost always means params were captured by a
   jit closure (law 3) or the learning rate is wrong for the init.
3. No `Referenced tracer ... freed` in the console after 200+ steps.
4. Memory is flat: log `np.Array` counts or watch the tab's memory across 1000
   steps. A rising line means a missing `.dispose()` in a per-step path.
5. The page still scrolls at 60fps while training.
6. Non-WebGPU browsers get a real fallback (wasm for small models, honest prose
   for large ones) — not a dead button.

## Version notes

Written against **`@jax-js/jax` 0.1.21** and **`@jax-js/optax` 0.1.2**
(August 2026). Re-run `scripts/doctor.mjs` on upgrade; it asserts the specific
behaviours the laws above depend on.

- 0.1.19 fixed eager-mode `grad` of matmul materialising an M×K×N intermediate,
  and added gather/sort backprop via `scatter` — **eager only**; law 4 still
  holds under `jit`.
- optax has not been republished since January 2026, so the in-graph Adam bias
  correction landed upstream is not yet on npm.

Docs: <https://jax-js.com/docs/> · Repo: <https://github.com/ekzhang/jax-js> ·
Feature matrix: `FEATURES.md` in that repo.
