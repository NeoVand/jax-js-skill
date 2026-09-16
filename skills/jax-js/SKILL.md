---
name: jax-js
description: Build, train, evaluate and debug neural networks in the browser with jax-js (@jax-js/jax) and @jax-js/optax on WebGPU or WebAssembly. Use for JaxJS APIs, autodiff and jit, array ownership, worker-based training, browser ML demos, transformers, autoencoders, latent world models and learned control. Includes tested version-specific workarounds, responsive interfaces, and checks that distinguish falling loss from useful learning.
license: MIT
---

# jax-js — training in the browser

jax-js provides JAX-style arrays, autodiff, `jit`, `vmap` and pytrees in
TypeScript. Parameters are plain trees of arrays; you differentiate a loss
function. Training and inference can stay entirely on the user's device.

**Verified 2026-09-16 with `@jax-js/jax` 0.1.24 and `@jax-js/optax` 0.1.2.**
Inspect the target project's installed versions first. Do not silently upgrade
an existing application to match this skill. Run the doctor after upgrades;
release notes and unreleased source are not substitutes for checking npm builds.

## Setup

```bash
npm i @jax-js/jax @jax-js/optax
```

```ts
import { init, defaultDevice, numpy as np, nn, jit, valueAndGrad, tree } from '@jax-js/jax';
import { adam, applyUpdates } from '@jax-js/optax';

const available = await init();
const device = available.includes('webgpu') ? 'webgpu'
  : available.includes('wasm') ? 'wasm' : 'cpu';
defaultDevice(device);
```

Initialize before creating arrays, inside the worker that owns them. Use CPU
for small diagnostics; wasm can outperform WebGPU on small workloads. Measure
the actual shapes and device before promising interactive training. See
[performance.md](references/performance.md) for benchmark methodology and
explicitly historical measurements.

Vite workers need ES-module output because jax-js lazily imports backends:

```ts
// vite.config.ts
import { defineConfig } from 'vite';
export default defineConfig({
  worker: { format: 'es' },
  optimizeDeps: { include: ['@jax-js/jax', '@jax-js/optax'] },
});
```

Always check a production build: dev can work while the default `iife` worker
format fails. Feature-detect WebGPU in a secure context (HTTPS or localhost),
then check actual initialization; `navigator.gpu` alone is not proof of a usable
adapter. Put a deadline around boot and provide a retry or a practical fallback.

## Five core rules

### 1. Array operations consume inputs. Use `.ref` for another use.

```ts
const x = np.array([1, 2, 3]);
np.sum(x.ref).item(); // extra use
np.sum(x).item();     // final use
// np.sum(x) now throws: the array was consumed
```

Methods consume receivers and array operands (`x.add(y)` consumes both).
Count uses, lend `.ref` on all but the last, and dispose unused results.
`tree.ref(params)` retains every leaf; `tree.dispose(params)` releases them.
Metadata (`shape`, `size`, `dtype`) and tree traversal (`leaves`, `flatten`)
do not consume arrays; a `tree.map` callback may.

**`.ref` is neither a buffer copy nor a gradient detach.** Use
`lax.stopGradient` only where the objective calls for it. A target branch is not
automatically detached just because it is called a target. Details and leak
patterns: [memory.md](references/memory.md).

### 2. Reading a value consumes it. Never dispose the same use afterward.

`.item()`, `.js()`, `.dataSync()` and `await .data()` all consume.
`blockUntilReady()` does not.

```ts
const preview = lossVal.ref.item(); // retained lossVal remains usable
const final = lossVal.item();       // consumes its final use; no dispose
```

If the loss is only sometimes read, dispose it in the other branch:

```ts
if (step % 50 === 0) log(lossVal.item());
else lossVal.dispose();
```

### 3. Jit hot paths, keep shapes stable, pass changing values as arguments.

`jit` caches traces by input signature, including shape/dtype and static values.
Pass params explicitly; closure-captured arrays are frozen at trace time:

```ts
const forward = jit((p, tok, pos) => model(p, tok, pos));
forward(tree.ref(params), tok, pos);
```

Pad variable-length prompts to a fixed block size and read the last real token's
logits. Right padding does not affect earlier rows under causal attention;
exclude padding from losses. Use eager execution for debugging when useful.

`staticArgnums` retraces per distinct value: do not put step counters or frequently
changing learning rates there. Pass dynamic scalars as arrays when needed.
Dispose owned jitted functions at teardown.

### 4. Verify differentiated indexing; use one-hot embeddings where needed.

On 0.1.24, the tested `jit(grad(...np.take...))` path still fails with
`routine primitive scatter input is not imm`. Eager gather gradients work.
The template's compiled embedding path uses:

```ts
const tokenOH = nn.oneHot(inputIds, vocab); // [B, S, V]
const x = np.dot(tokenOH.reshape([-1, vocab]), params.wte); // [B·S, D]
```

Build one-hots outside the step as the templates do. Budget `4·B·S·V` bytes
for each float32 one-hot tensor; inputs and targets may both need one.
This workaround is version-specific, not an architectural requirement.
`np.take` remains suitable for inference-only or non-differentiated paths.

### 5. Give training a worker and an explicit lifecycle.

Use a worker for sustained training so dispatch, compilation and readbacks do
not block the UI. Yield between short bursts so stop requests can arrive.
**Async message handlers can overlap across `await`**: serialize model operations
and let stop set a cancellation flag outside that queue.

One worker is sufficient when inference runs between bursts or while paused.
Use a second worker for expensive concurrent inference only when measurements
justify duplicated weights, caches, buffers and checkpoint traffic. Label its
outputs with the checkpoint step they actually used.

Disposal must stop accepting work, reject pending RPCs and terminate the worker
after a bounded graceful attempt. Generation counters prevent late results from
writing into a reset or unmounted UI. See [workers.md](references/workers.md).

## Training step

**Default: jit loss and gradients; use Optax Adam outside jit.**

```ts
const jitStep = jit((p, x, y) =>
  valueAndGrad((pp) => lossFn(pp, x, y))(p));
const solver = adam(3e-4, { b1: 0.9, b2: 0.99 });
let optState = solver.init(tree.ref(params));

for (let i = 0; i < steps; i++) {
  const { x, y } = nextBatch(); // fresh owned arrays
  const [lossVal, grads] = jitStep(tree.ref(params), x, y);
  const [updates, nextState] = solver.update(grads, optState, tree.ref(params));
  params = applyUpdates(params, updates);
  optState = nextState;
  const loss = lossVal.item(); // consumes; no dispose
  record(loss);
}
```

`@jax-js/optax@0.1.2` Adam calls `count.item()` for bias correction, so the
compiled path throws `count.item is not a function`. This is a published-package
limitation; an upstream source fix does not update an already installed package.
Other transformations need their own compatibility checks.

If optimizer dispatch is a measured bottleneck, use `templates/fused-adam.ts`.
Its device-scalar bias corrections avoid per-step retracing, and its results
are tested against Optax within floating-point tolerances. Match the intended
optimizer's clipping, schedules and decay before replacing it. Plain SGD is
also appropriate for small teaching examples.

Changing Adam's learning rate need not reset moments: retain compatible
`optState` when rebuilding the transformation, or use a schedule. Calling
`solver.init` resets state; loading a weights-only checkpoint does not restore it.

### Pacing and measurement

Start with a scalar loss read each step and yield every few updates:

```ts
if (stopRequested) break;
trainStep();
if (i % 4 === 3) await new Promise((resolve) => setTimeout(resolve, 0));
```

Benchmark sync frequency on the target. A historical run found per-step sync
faster than every ten steps, but that is not a universal GPU rule. Time completed
work, separate compile time from steady state, and report the backend. Keep
training progress independent of `requestAnimationFrame`, which pauses in hidden
tabs. Coalesce UI paints, not the worker's ability to receive cancellation.

## Choose the app structure

| Situation | Starting point |
| --- | --- |
| Tiny teaching script with bounded work | `templates/standalone-lab.ts`; yield and measure responsiveness |
| Sustained training | `templates/worker.ts` + `templates/engine.ts` |
| Expensive inference concurrent with training | `templates/twin-engine.ts`; compare against one worker |
| Framework UI | Component/page-owned engine; reactive metrics, explicit cleanup. [ui.md](references/ui.md) |
| World model or learned control | [world-models.md](references/world-models.md) before defining losses or displays |

## Reference index

Read only the files needed for the task.

| I need to… | Read |
| --- | --- |
| Look up operators, dtypes, random keys or shape rules | [api.md](references/api.md) |
| Debug ownership, leaks or freed arrays | [memory.md](references/memory.md) |
| Structure RPC, lifecycle and optional sampler workers | [workers.md](references/workers.md) |
| Write an MLP, CNN, transformer, autoencoder or VAE | [models.md](references/models.md) |
| Build policy-gradient or preference-training demos | [rl.md](references/rl.md) |
| Teach with charts, controls, canvases and visible learning evidence | [ui.md](references/ui.md) |
| Build and evaluate a latent world model | [world-models.md](references/world-models.md) |
| Measure speed and resource costs | [performance.md](references/performance.md) |
| Diagnose errors or stalled learning | [troubleshooting.md](references/troubleshooting.md) |

## Templates and scripts

Adapt the templates to the target application; UI examples need the documented
host imports. Preserve ownership and lifecycle invariants, not arbitrary sizes
or styling. API tests exercise model math; browser tests exercise worker demos.

| File | Purpose |
| --- | --- |
| `templates/standalone-lab.ts` | Small complete training loop and canvas |
| `templates/worker.ts` | Model owner, serialized RPC operations and stop |
| `templates/engine.ts` | Promise RPC client with streaming metrics |
| `templates/twin-engine.ts` | Optional trainer/sampler pair and checkpoint courier |
| `templates/tokens.ts` | Validate token ranges and context length |
| `templates/model-mlp.ts` | Configurable MLP, MSE/cross-entropy and VAE bottleneck |
| `templates/model-transformer.ts` | Decoder, sampling, attention and residual capture |
| `templates/fused-adam.ts` | Adam fused into a jitted step |
| `templates/ui-svelte5.svelte` | Svelte lifecycle, loss chart and controls |
| `templates/ui-react.tsx` | React lifecycle, loss chart and controls |

From the target project's directory, use the installed skill's actual path:

```bash
node /path/to/jax-js/scripts/doctor.mjs
node /path/to/jax-js/scripts/scaffold.mjs <new-directory>
```

## Before saying it works

1. Run the doctor on installed packages and relevant numerical contract tests.
2. Show training loss **and** fixed held-out evidence against a simple baseline.
   Use matched seeds/data/checkpoints for comparisons. Falling loss alone can
   hide representation collapse, leakage or a stale inference model.
3. Check the real browser: train, pause, infer, reset, change settings, leave
   during boot/training and revisit. Check errors and actual model behavior.
4. Inspect long-run resource use and repeated teardown. Backend pools and jit
   caches may grow during warm-up; investigate continued growth after shapes
   stabilize. Audit array ownership rather than assuming JS GC releases it.
5. Verify a production build, practical backend fallback, and responsive controls.
6. Inspect the visual evidence in light/dark themes and at narrow widths. Diagrams,
   equations, model inputs and captions must describe the computation that runs.

## Version notes

0.1.22 fixes integer/boolean `mean` truncation and adds numerical helpers;
0.1.23 adds APIs and changes Firefox completion polling; 0.1.24 adds `select`,
`polyder` and NaN-aware operations. See [api.md](references/api.md) for details.
The gather-gradient and published Adam workarounds were re-tested on 0.1.24 /
0.1.2, not inferred from release notes.

Primary sources: [releases](https://github.com/ekzhang/jax-js/releases),
[docs](https://jax-js.com/docs/),
[feature matrix](https://github.com/ekzhang/jax-js/blob/main/FEATURES.md).
