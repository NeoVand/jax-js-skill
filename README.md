# jax-js skill

An [agent skill](https://skills.sh) for building and training **real neural
networks in the browser** with [jax-js](https://github.com/ekzhang/jax-js) —
JAX's semantics (autodiff, `jit`, `vmap`, pytrees, optax) in TypeScript, running
on WebGPU.

```bash
npx skills add NeoVand/jax-js-skill
```

Works with Claude Code, Cursor, Codex, Copilot, Windsurf, Cline, Gemini and the
other 70-odd agents the `skills` CLI supports.

## What it knows

jax-js is not TensorFlow.js. There is no `model.fit()`, no layer objects, and —
the thing that breaks everyone — **arrays are moved, not shared**. An agent
writing jax-js from general knowledge produces code that throws
`Referenced tracer ... freed` on the second line. This skill front-loads the
ownership model, then gets specific:

- **Five laws** that prevent the failures: ownership and `.ref`, readback
  consuming, `jit` shape caching and the closure trap, one-hot embeddings, and
  worker architecture.
- **Architecture** — a worker that owns the model, a promise RPC on the main
  thread, and the *twin-worker courier*: a second worker that answers the UI
  from couriered checkpoints so training never pauses to draw a sample.
- **Model recipes** — MLP, autoencoder, VAE, decoder-only transformer with
  attention and residual-stream capture, REINFORCE/GRPO, DPO.
- **UI patterns** — lifecycle phases, boot-on-scroll, coalesced rendering,
  charts, canvas, and the Svelte 5 / React versions of each.
- **Measured performance guidance**, not folklore. See below.

## What it found

Everything the skill asserts is verified by `tests/api.test.mjs` (38 assertions,
runs in Node on cpu/wasm) and measured by `examples/bench.html` (a real browser,
a real GPU). Some results were surprising:

| Finding | Measured |
| --- | --- |
| Fusing the optimizer into `jit` instead of running optax outside it | **2.9× faster** on a 235k-param transformer on WebGPU; up to 6× on small MLPs |
| `grad` through `np.take` inside `jit` | still throws at 0.1.21 — embeddings must be one-hot matmuls |
| `@jax-js/optax@0.1.2` inside `jit` | impossible: `treeBiasCorrection` reads its counter back to the host |
| `.item()` / `.js()` / `.dataSync()` / `await .data()` | **all consume** the array — `x.item(); x.dispose();` is a double free |
| wasm vs WebGPU below ~100k params | wasm **ties or wins** — dispatch latency dominates |
| WebGPU at 5.3M params | 10× faster than wasm (56 ms/step — still interactive) |
| Syncing the loss every step vs every 10 | every step is **1.8× faster** (jax-js issue #151 reproduces) |

## Layout

```
skills/jax-js/
  SKILL.md              the five laws, the canonical step, routing
  references/           api · memory · workers · models · rl · ui · performance · troubleshooting
  templates/            runnable: worker, engine, twin-engine, transformer, MLP, fused Adam, Svelte, React
  scripts/
    doctor.mjs          re-verify every assumption against the installed version
    scaffold.mjs        write a runnable vite + worker starter
examples/               two working demos + the benchmark suite
tests/                  the contract tests
```

## Verifying it yourself

```bash
npm install
npm test                       # 38 API + template contract assertions, in Node
node skills/jax-js/scripts/doctor.mjs
cd examples && npm install && npm run dev      # /sine.html /gpt.html /bench.html
```

`doctor.mjs` is the important one on upgrade: jax-js moves fast, and two of the
five laws exist only because of gaps that may close. Some of its failures are
good news, and it says so.

## Provenance

The architecture comes from [jaxverse](https://github.com/NeoVand/jaxverse), an
interactive book that trains eight real models in the browser — including the
twin-worker courier, which is why its loss curves do not stutter while the model
is writing.

MIT.
