# Performance

All numbers below were measured with the skill repo's
[examples/bench.html](https://github.com/NeoVand/jax-js-skill/blob/main/examples/bench.html) in Chrome 148 on an
Apple Silicon Mac (Metal-3 adapter, hardware — not a fallback). **Run it on your
own machine before quoting any of them**; absolute times vary by an order of
magnitude between GPUs, and the first run of anything is dominated by shader
compilation and GPU clock-up. The *ratios* are what transfer.

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

Use `templates/fused-adam.ts`. It converges bit-for-bit with optax's Adam
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

Conclusions that hold across runs:

- **`cpu` is unusable** for anything but a scalar demo — 100–200× slower than
  wasm. Never `defaultDevice('cpu')` deliberately.
- **`webgl` is a distant third.** Treat it as a last resort, not a fallback tier.
- **wasm is genuinely good** — it matches OpenBLAS on Apple Silicon — and it
  *ties or beats WebGPU below ~100k parameters*, where dispatch latency dominates.
  So a small MLP demo needs no GPU at all, and a no-WebGPU fallback for one is
  honest rather than a consolation prize.
- **WebGPU pulls away fast above ~250k parameters**: 1.8× at 0.24M, 8.6× at
  1.9M, 10× at 5.3M. A 5M-parameter transformer at 56 ms/step is interactive.

Practical fallback chain: `webgpu → wasm`, and tell the user which they got.

## 3. Sync every step. Do not let the queue run ahead.

Intuition says batching GPU work and blocking once at the end should be faster.
It is not — measured at 0.1.21, on every backend:

| Backend | sync every step | sync every 10 steps |
| --- | --- | --- |
| wasm | **43.9 ms** | 67.4 ms |
| webgl | **173 ms** | 182 ms |
| webgpu | **32.4 ms** | 58.4 ms |

Reading the loss every step is ~1.8× *faster* than letting ten steps queue up.
This reproduces [jax-js issue #151](https://github.com/ekzhang/jax-js/issues/151),
which is closed for want of a reduction rather than fixed. So: read the loss
every step, and use burst size for pacing instead.

## 4. Where the rest of the time goes

**jit compilation.** The first call with a given shape signature compiles. That
is seconds for a transformer. It is also why changing shapes is so expensive —
pad instead. Warm the compile during boot (run one step before showing "ready")
so the user's first click is not the slow one.

**One-hot embeddings.** `nn.oneHot(ids, V)` for a batch is `B·S·V` floats:
8 × 96 × 24 is trivial, 8 × 256 × 8000 is 16M floats per step and will hurt.
This is the real ceiling on vocabulary size in the browser, and the reason
character and small word-piece vocabularies dominate in-browser LMs. Build the
one-hots outside the jitted function so they are not retraced.

**Readback.** `dataSync()` blocks the thread until the GPU drains. On the main
thread that is a dropped frame; in a worker it is fine. Read only what you draw:
one loss scalar per step, not the whole logits tensor.

**Transfers.** Move `ArrayBuffer`s between worker and page with the transfer
list, never by structured clone. A 20 MB checkpoint copies in ~10 ms and
transfers in ~0.

## 5. Making it *feel* fast

Wall-clock is not the whole story. Two structural moves matter more than any
micro-optimisation:

- **Train in a worker.** Measured faster than the main thread, and the page
  keeps painting. See [workers.md](workers.md).
- **Sample from a second worker.** Otherwise every specimen the UI asks for
  stops training for its duration, and the loss curve stutters. See
  [the twin-worker courier](workers.md#the-twin-worker-courier--the-big-win).

Then pace deliberately: bursts of 25–50 steps, an eval between bursts, one paint
per animation frame. A model that steps at 10 ms but repaints the DOM 100 times
a second is slower than one that steps at 15 ms and repaints 60 times.

## 6. A checklist when something is slow

1. Is the optimizer inside the jitted step? (up to 6×)
2. Is anything recompiling? Log the trace count, or watch for a step that is
   1000× the median. Fix by padding to fixed shapes.
3. Are you reading back more than one scalar per step?
4. Is the batch one-hot bigger than the model?
5. Is the model big enough to deserve WebGPU at all? Below ~100k parameters,
   wasm is faster.
6. Is the UI re-rendering per step instead of per frame?
7. Is the first-step compile being counted in your average? Report medians after
   warm-up.
