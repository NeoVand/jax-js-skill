# Performance

Historical measurements below used **jax-js 0.1.21 + optax 0.1.2**, before the
current 0.1.24 verification. They were measured with the skill repo's
[examples/bench.html](https://github.com/NeoVand/jax-js-skill/blob/main/examples/bench.html) in Chrome 148 on an
Apple Silicon Mac (Metal-3 adapter, hardware — not a fallback). **Run it on your
own machine before quoting any of them**; absolute times vary by an order of
magnitude between GPUs, and the first run of anything is dominated by shader
compilation and GPU clock-up. Ratios also depend on hardware, browser, shapes
and package versions; these tables are hypotheses to test, not thresholds.

```bash
cd examples && npm install && npm run dev   # then open /bench.html
```

## 1. Fuse the optimizer into `jit` — the biggest easy win

Running optax outside the jitted step means every parameter tensor gets its own
handful of tiny kernel dispatches per update (two moments, bias correction, the
update, the add). On a 6-tensor MLP that is ~30 round-trips of pure latency.
Inside one traced graph they fuse.

| Model | Backend | optax outside `jit` | fused inside `jit` | speedup |
| --- | --- | --- | --- | --- |
| MLP [64,128,128,64] | wasm | 3.7 ms | 0.6 ms | **6.2×** |
| MLP [64,128,128,64] | webgpu | 7.9 ms | 1.9 ms | **4.2×** |
| transformer 235k params | wasm | 38.4 ms | 28.1 ms | 1.4× |
| transformer 235k params | webgpu | 26.7 ms | **9.2 ms** | **2.9×** |

The win grows as the model gets *smaller relative to the number of tensors*,
because it is latency, not arithmetic. But even on a real transformer it is
nearly 3× on WebGPU.

Use `templates/fused-adam.ts`. Its numerical results match optax's Adam within the test's floating-point tolerances
(asserted in [tests/api.test.mjs](https://github.com/NeoVand/jax-js-skill/blob/main/tests/api.test.mjs)). Keep optax when you need its schedules,
chains or weight decay and the step is already long.

## 2. Pick the backend by model size, not by reflex

| Workload | cpu | wasm | webgl | webgpu |
| --- | --- | --- | --- | --- |
| MLP [2,64,64,2], 256 rows | 1168 ms | **5.1 ms** | 41 ms | 8.2 ms |
| transformer 235k params | (minutes) | 37.9 ms | 171 ms | **26.9 ms** |

And with the fused optimizer, as the model grows:

| Transformer | params | wasm | webgpu |
| --- | --- | --- | --- |
| 2×96, V=24 | 0.24M | 27 ms | **15 ms** |
| 4×192, V=256 | 1.89M | 214 ms | **25 ms** |
| 6×256, V=1024 | 5.28M | 570 ms | **56 ms** |

These runs show that small workloads can favor wasm while larger matrix
operations benefit from WebGPU. Parameter count alone does not determine the
crossover: batch size, sequence length, tensor count and dispatch overhead also
matter. Benchmark the actual model; keep CPU for small diagnostics and tests.

A practical starting fallback is WebGPU → wasm → CPU (only if the workload is
small enough). Report the selected device and avoid promising real-time speed
before measurement.

## 3. Measure synchronization cadence

Intuition says batching GPU work and blocking once at the end should be faster.
The opposite happened in this 0.1.21 benchmark:

| Backend | sync every step | sync every 10 steps |
| --- | --- | --- |
| wasm | **43.9 ms** | 67.4 ms |
| webgl | **173 ms** | 182 ms |
| webgpu | **32.4 ms** | 58.4 ms |

Reading the loss every step is ~1.8× *faster* than letting ten steps queue up.
This reproduces [jax-js issue #151](https://github.com/ekzhang/jax-js/issues/151),
which is now closed as not planned. That status does not establish behavior on
every device or release. Start with a per-step scalar readback for simple
bounded execution; benchmark less frequent synchronization on the target.
Always await actual completion when timing, consume or dispose every loss, and
yield so stop messages can be processed. Version 0.1.23 also changed Firefox
completion polling, another reason to remeasure.

## 4. Where the rest of the time goes

**jit compilation.** The first call with a given shape signature compiles. That
is seconds for a transformer. It is also why changing shapes is so expensive —
pad instead. Expose a "compiling" phase or warm a representative call during boot. If a
warm-up updates weights or optimizer state, count it as training or restore all
state afterward; the displayed untrained baseline must really be untrained.

**One-hot embeddings.** `nn.oneHot(ids, V)` for a batch is `B·S·V` floats:
8 × 96 × 24 is trivial, 8 × 256 × 8000 is 16M floats per step and will hurt.
Budget this explicitly along with gradients and activations. Building one-hots
outside `jit` matches these templates; it is the input signatures, not where
one-hot is called, that determine retracing.

**Readback.** `dataSync()` blocks the thread until the GPU drains. On the main
thread that is a dropped frame; in a worker it is fine. Read only what you draw:
one loss scalar per step, not the whole logits tensor.

**Transfers.** Move `ArrayBuffer`s between worker and page with the transfer
list, never by structured clone. Transfer avoids the structured-clone buffer copy; GPU readback and upload still cost time.

## 5. Making it *feel* fast

Wall-clock is not the whole story. Two structural moves matter more than any
micro-optimisation:

- **Train in a worker.** Measured faster than the main thread, and the page
  keeps painting. See [workers.md](workers.md).
- **Consider a second worker for expensive concurrent inference.** Compare the
  added memory and checkpoint traffic with pausing one worker between bursts. See
  [the twin-worker courier](workers.md#the-twin-worker-courier--optional-concurrent-inference).

Then pace deliberately: bursts of 25–50 steps, an eval between bursts, one paint
per animation frame. A model that steps at 10 ms but repaints the DOM 100 times
a second is slower than one that steps at 15 ms and repaints 60 times.

## 6. A checklist when something is slow

1. Is the optimizer inside the jitted step? (up to 6×)
2. Is anything recompiling? Log the trace count, or watch for a step that is
   1000× the median. Fix by padding to fixed shapes.
3. Are you reading back more than one scalar per step?
4. Is the batch one-hot bigger than the model?
5. Have you compared wasm and WebGPU on the actual workload?
6. Is the UI re-rendering per step instead of per frame?
7. Is the first-step compile being counted in your average? Report medians after
   warm-up.
