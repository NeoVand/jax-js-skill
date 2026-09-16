# Building the interface

A training UI has one job the rest of the app does not: it must stay honest and
stay smooth while something expensive runs underneath it.

## The lifecycle contract

Explicit phases, one variable. Every control reads from it; nothing infers.

```ts
type Phase = 'idle' | 'loading' | 'ready' | 'training' | 'error' | 'no-webgpu';
```

- **idle** — nothing built yet. Show the frame and a quiet placeholder.
- **loading** — with a *specific* note ("fetching the corpus (1.5 MB)…",
  "building the model on your GPU…"), not a bare spinner.
- **ready** — the first button the user sees is one that *does* something
  (Train, Play, Sample). Never a "Load" button; boot on demand.
- **training** — the same button becomes Pause. Live numbers in the header.
- **error** — the real message plus a Retry. Give every boot step a deadline
  (see [workers.md](workers.md#generation-counters-and-deadlines)) so a stalled fetch
  becomes an error, not an eternal spinner.
- **no-webgpu** — a real fallback. Small models run on wasm; large ones get
  honest prose, a recorded animation, or a screenshot. Not a dead button.

Probe capabilities before enabling GPU-dependent controls; select the actual
backend during boot. Show a loading state while that decision is pending.

## Separate resource handles from render state

Keep the worker/engine in a component-owned plain field (Svelte) or ref (React),
and expose metrics and phase as reactive values. This makes ownership and cleanup
explicit. React `useState` does not proxy objects, and Svelte `$state` does not
deeply proxy class instances; neither universally breaks `postMessage`.

A Svelte plain-object state proxy is not structured-cloneable. Send a plain
payload, a `$state.snapshot` of serializable data, or a copied typed array; do
not send framework proxies or device arrays across the worker boundary.

Share a page-owned lab through context when several components need it. Avoid
mutable server-side module singletons, which can be shared between SSR requests.
See [Svelte state](https://svelte.dev/docs/svelte/$state) and
[SvelteKit state management](https://svelte.dev/docs/kit/state-management).

## Boot when the user gets there, not on mount

Prerendered pages and long documents mean the model may never be needed. Boot on
intersection, ~160px before the demo enters the viewport, once:

```ts
export function inview(node: HTMLElement, fire: () => void) {
  const io = new IntersectionObserver(
    (entries) => { if (entries[0].isIntersecting) { io.disconnect(); fire(); } },
    { rootMargin: '160px' }
  );
  io.observe(node);
  return { destroy: () => io.disconnect() };
}
```

## Coalesce renders

A metrics callback fires every step. Rendering on each one makes the UI the
bottleneck. Coalesce to one paint per frame:

```ts
let queued = 0;
const notify = () => {
  if (queued) return;
  queued = requestAnimationFrame(() => { queued = 0; render(); });
};
```

`requestAnimationFrame` does not fire in a hidden tab — which is correct for
painting, and a trap for anything else. Never drive training progress, timeouts
or e2e signals from rAF.

## Framework lifecycle

`templates/ui-svelte5.svelte` uses runes for metrics and a plain engine field.
`templates/ui-react.tsx` uses state for metrics and refs for resource handles.
Adapt their imports and styling to the host app.

- Create workers only in browser lifecycle paths, not during SSR or render.
- Assign the engine handle before awaiting initialization, so cleanup can reach
  an engine that is still booting. Dispose it on initialization failure as well.
- Guard every asynchronous result and streamed metric with the current engine
  identity or generation. An old boot, evaluation or training loop must not
  publish into a new run.
- In React development StrictMode, effects get an extra setup/cleanup cycle;
  keep each effect's engine local and dispose that specific engine in cleanup.
- In Svelte, `onDestroy` can run during server rendering; keep cleanup safe when
  no worker was created. A reactive class can live in a `*.svelte.ts` module,
  while its instance belongs to the page/context that owns its lifetime.
- Coalesce paints, and cancel pending animation-frame callbacks on teardown.

The cancellation and RPC rules are in [workers.md](workers.md). A successful
initialization after unmount is still a resource leak unless it is disposed.

## Charts

SVG for loss curves, canvas for anything with thousands of points.

- Use a labeled log scale when positive loss spans orders of magnitude. Handle
  zero/non-finite values explicitly; use a linear or suitable signed scale for
  objectives that can be negative. Keep comparison axes consistent.
- Train curve in the accent colour, validation in the contrast colour, and plot
  validation as points-plus-line at burst boundaries — it is measured less often
  and pretending otherwise is a lie.
- Hairline gridlines, tabular-numeral labels at 10–11px, unobtrusive axes.
- Fixed `viewBox` with `preserveAspectRatio="none"` and string-built paths is
  enough for a live curve; no charting library required.

[examples/src/chart.ts](https://github.com/NeoVand/jax-js-skill/blob/main/examples/src/chart.ts) in the skill repo is a
complete 60-line implementation.

## Canvas

```ts
const dpr = Math.min(2, window.devicePixelRatio || 1);   // cap at 2
const W = canvas.clientWidth, H = 220;
if (canvas.width !== W * dpr) { canvas.width = W * dpr; canvas.height = H * dpr; }
ctx.setTransform(dpr, 0, 0, dpr, 0, 0);                  // every frame
ctx.clearRect(0, 0, W, H);
```

Read colours from CSS custom properties at draw time. If the canvas is not
continuously animating, redraw when the theme changes:

```ts
const token = (name: string, fallback: string) =>
  getComputedStyle(canvas).getPropertyValue(name).trim() || fallback;
ctx.strokeStyle = token('--accent', '#2b45d8');
```

Honour `matchMedia('(prefers-reduced-motion: reduce)')`: no autoplaying
animation; render a meaningful static frame instead.

## Controls

- **Transport in the header** (Train/Pause, Step, Reset). Parameters below the
  stage or in a side column. Keep the whole demo inside one viewport.
- **Every control must change something visible.** If a slider would not visibly
  move the demo, cut it.
- Match existing app controls, typography and spacing. Use a pace tuned for
  reading; add a speed control only if it helps the lesson.
- Status line shows live truth: `step 240 · loss 0.312 nats · 12 ms/step`, in
  tabular numerals so digits do not jitter.
- Disable, do not hide. A control that vanishes mid-run is disorienting.

## Token input contract

The language-model templates accept validated token IDs through
`templates/tokens.ts`. Check integer values, vocabulary bounds and context
length before building one-hots; an out-of-range ID can produce an all-zero row
instead of a useful error. Recheck at the worker entry point.

Encoding may live in the app or worker as appropriate. Encoding text as numbers
does not make its content trusted. Validation prevents malformed tensors and
vocabulary mismatches; it is not a prompt-injection defense.

## Never render model output as HTML

A sampler emits whatever the weights make likely, and once a user can type the
prompt, they choose part of that. Put it on the page as **text**, never as
markup:

```ts
// ✓ escaped
el.textContent = sample.text;
// ✗ an XSS sink fed by a text generator
el.innerHTML = sample.text;
```

Same rule in every framework: `{text}` in Svelte and React is escaped and is
what you want; `{@html text}` and `dangerouslySetInnerHTML` are not. If the demo
genuinely needs to render generated markdown, sanitise it — do not hand raw
model output to a markdown renderer with HTML passthrough enabled.

## Show evidence of learning

Keep training loss visible, with units and a useful baseline. Pair it with a
fixed held-out test and a directly inspectable before/after example. The metric
must test the claim the reader is meant to learn: classification accuracy,
next-frame discrimination, a rollout versus persistence, or executed goal error.
Do not replace the loss with an unexplained success badge.

Choose one clear question for each plate and one intervention that answers it.
For example: same history, different actions; same model, longer horizon; same
initial weights/data, with and without a regularizer. A compact architecture
SVG can connect what enters the model, what is learned, and what the loss
compares. Reuse the same symbols and colors in diagrams, equations and controls.
Inspect the rendered SVG, including subscripts, labels and arrow endpoints.

Long comparisons need visible progress: show the phase, completed updates and
partial evidence as it becomes available. Label old results with their checkpoint
step while new evaluation is running. Keep stop/reset responsive and avoid a
long blank stage followed by everything appearing at once.

Separate training data, held-out evaluation and any fitted visualization readout.
A decoder or renderer can hide model failures; disclose what is actually learned
and what is supplied by geometry or labels. See [world-models.md](world-models.md).

Changing learning rate does not inherently reset Adam. Describe the actual
behavior of the implementation. Weights-only checkpoint loading usually does
reset optimizer history; label that when relevant.

For image inputs, verify crop and orientation against the main widget. A themed
preview can use display-only inversion in dark mode without changing the raw
pixels fed to the model. Preview enlargement is not increased sensor resolution.
Check light/dark, narrow layouts, reduced motion and the app's existing visual
language in a real browser.

## Reset that actually resets

Keep the step-0 checkpoint from boot and reload it, rather than re-initialising:

```ts
this.initCkpt = await engine.exportCheckpoint();       // at boot
await engine.loadWeights(this.initCkpt.slice(0));      // on reset; slice keeps the master
```

First stop/serialize active work and invalidate pending results. Then clear
curves, samples and derived readouts together. A weights reset does not reset
random generators, data order or optimizer history unless those are explicitly
restored. Preserve all of them if the UI promises an identical replay.
