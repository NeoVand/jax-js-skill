# jax-js skill

[![skills.sh](https://img.shields.io/badge/skills.sh-jax--js-1f6feb)](https://skills.sh/neovand/jax-js-skill)
[![license](https://img.shields.io/badge/license-MIT-555)](LICENSE)

<!-- skills.sh also serves a live install-count badge, but it renders a literal
     "resource not found" image while the count is still 0 — the directory page
     exists and lists the skill, the counter just has not accumulated yet.
     Swap the static badge above for this one once it shows a number:
[![skills.sh](https://skills.sh/b/neovand/jax-js-skill)](https://skills.sh/neovand/jax-js-skill)
-->

An [agent skill](https://skills.sh) for building and training **real neural
networks in the browser** with [jax-js](https://github.com/ekzhang/jax-js) —
JAX's semantics (autodiff, `jit`, `vmap`, pytrees, optax) in TypeScript, running
on WebGPU.

```bash
npx skills add NeoVand/jax-js-skill
```

Install through the `skills` CLI for supported coding agents, or read
[SKILL.md](skills/jax-js/SKILL.md) directly.

## What it knows

jax-js is not TensorFlow.js. There is no `model.fit()`, no layer objects, and —
the thing that breaks everyone — **arrays are moved, not shared**. An agent
writing jax-js from general knowledge produces code that throws
`Referenced tracer ... freed` on the second line. This skill front-loads the
ownership model, then gets specific:

- **Five core rules** covering: ownership and `.ref`, readback
  consuming, `jit` shape caching and the closure trap, one-hot embeddings, and
  worker architecture.
- **Architecture** — a worker that owns the model, a promise RPC on the main
  thread, serialized model operations, bounded disposal, and an optional second
  worker for concurrent inference when its memory and transfer costs are justified.
- **Model recipes** — MLP, autoencoder, VAE, decoder-only transformer with
  attention and residual-stream capture, REINFORCE, group-relative advantages
  (distinguished from full GRPO), and DPO.
- **UI patterns** — lifecycle phases, boot-on-scroll, coalesced rendering,
  charts, canvas, and the Svelte 5 / React versions of each.
- **World models and learning evidence** — action/history alignment, gradient
  paths, collapse checks, held-out baselines, faithful rollouts and planning.
- **Performance guidance** separating measured local results from API contracts.

## What it found

Verified **2026-09-16** against **jax-js 0.1.24** and **optax 0.1.2**. The
Node suite checks numerical/ownership contracts and RPC lifecycle behavior; the
browser suite runs training, concurrent sampling when WebGPU is available,
checkpoint replacement during a yielding train, and stop handling.

| Current finding | Guidance |
| --- | --- |
| Jitted `grad` through the tested `np.take` embedding path still fails | Keep the one-hot workaround and recheck on upgrade |
| Published Optax Adam 0.1.2 reads its counter on the host | Keep Adam outside `jit`, or use the tested fused implementation |
| `.item()`, `.js()`, `.dataSync()` and `.data()` consume arrays | Do not dispose the same use after reading it |
| `.ref` retains ownership without detaching gradients | Use `lax.stopGradient` only where the objective specifies it |
| `mean` on integer/boolean inputs was fixed in 0.1.22 | Fractional accuracy masks now reduce correctly |
| Optimizer state is explicit | Changing Adam's rate can preserve moments |

The older Chrome 148 / Apple Silicon benchmarks used **jax-js 0.1.21**. They
found benefits from optimizer fusion, wasm on small workloads, and per-step
synchronization. Those numbers are historical, not new 0.1.24 measurements or
universal hardware thresholds. See [performance.md](skills/jax-js/references/performance.md)
and rerun `/bench.html` on the target device.

## Layout

```
skills/jax-js/
  SKILL.md              core rules, the canonical step, routing
  references/           api · memory · workers · models · world-models · rl · ui · performance · troubleshooting
  templates/            runnable: worker, engine, twin-engine, transformer, MLP, fused Adam, Svelte, React
  scripts/
    doctor.mjs          probe critical assumptions against the installed version
    scaffold.mjs        write a runnable vite + worker starter
examples/               two working demos + the benchmark suite
tests/                  the contract tests
```

## Verifying it yourself

```bash
npm install
npm --prefix examples install
npm test                       # Node numerical/lifecycle tests + browser integration
npm run lint:skill             # frontmatter, size and link checks
npm --prefix examples run build # production worker bundling
node skills/jax-js/scripts/doctor.mjs
cd examples && npm install && npm run dev      # /sine.html /gpt.html /bench.html
```

`doctor.mjs` is the important one on upgrade: jax-js moves fast, and two of the
core rules contain workarounds for gaps that may close. Some of its failures are
good news, and it says so.

## Provenance

The architecture comes from [jaxverse](https://github.com/NeoVand/jaxverse), an
interactive book with browser-trained models, including language models and a
latent world-model chapter. Its experiments inform the worker, evaluation and
teaching patterns; they do not establish universal architecture rankings.

Library changes are checked against the [upstream releases](https://github.com/ekzhang/jax-js/releases)
and installed package code. The new world-model guidance distinguishes the
[LeWorldModel method](https://arxiv.org/abs/2603.19312v3) from the book's smaller
MLP adaptation.

MIT.
