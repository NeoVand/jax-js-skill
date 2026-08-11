# Examples

Three pages, all training real models in the tab. They exist to prove the
skill's templates run — and to be read.

```sh
npm install
npm run dev
```

| Page | What it is |
| --- | --- |
| `/sine.html` | `templates/standalone-lab.ts` — a small MLP fits a sine wave on the main thread. Runs on WebGPU, wasm or CPU. |
| `/gpt.html` | `templates/twin-engine.ts` — a 2-layer character transformer trained in one worker, with a **second** worker writing specimens from couriered checkpoints so the loss curve never pauses. Needs WebGPU. |
| `/bench.html` | The measurements behind `references/performance.md`. Takes a few minutes. |

The corpus for `/gpt.html` is generated in the page from a small grammar
(`src/corpus.ts`) — no downloads, and you can watch the model go from letter
frequencies to words to word order in about 600 steps.

The examples import the templates directly from `../skills/jax-js/templates/`,
so any edit to a template shows up here immediately. That is deliberate: if a
template breaks, these pages break.
