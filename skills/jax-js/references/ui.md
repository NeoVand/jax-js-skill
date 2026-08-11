# Building the interface

A training UI has one job the rest of the app does not: it must stay honest and
stay smooth while something expensive runs underneath it.

## The lifecycle contract

Five states, one variable. Every control reads from it; nothing infers.

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
  (see [workers.md](workers.md#deadlines-on-every-boot-step)) so a stalled fetch
  becomes an error, not an eternal spinner.
- **no-webgpu** — a real fallback. Small models run on wasm; large ones get
  honest prose, a recorded animation, or a screenshot. Not a dead button.

Probe for WebGPU on mount, before rendering any control, so the wrong affordance
never appears.

## Never put the engine in reactive state

The engine holds a `Worker` and is not serialisable, not cloneable, and not
something a framework should proxy. Keep it in a plain field.

```ts
// ✓ plain field
let engine: Engine | null = null;
// ✗ $state(new Engine(...)) / useState(engine) — the proxy will break postMessage
```

Only *derived numbers* belong in reactive state: step, loss, arrays of points,
sample strings.

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

## Svelte 5

```svelte
<script lang="ts">
  import { onDestroy } from 'svelte';
  import { Engine, detectWebGPU } from '$lib/engine';

  let phase = $state<'idle'|'loading'|'ready'|'training'|'error'|'no-webgpu'>('idle');
  let step = $state(0);
  let loss = $state(NaN);
  let curve = $state<Array<[number, number]>>([]);

  // NOT $state: a Worker handle must not be proxied.
  let engine: Engine | null = null;
  let playing = false;

  async function boot() {
    if (phase !== 'idle') return;
    phase = 'loading';
    if (!(await detectWebGPU())) { phase = 'no-webgpu'; return; }
    engine = new Engine({ tokenData });
    await engine.init(config);
    phase = 'ready';
  }

  async function toggle() {
    if (phase === 'training') { playing = false; await engine?.stop(); phase = 'ready'; return; }
    if (phase !== 'ready' || !engine) return;
    playing = true; phase = 'training';
    while (playing && engine) {
      await engine.train(40, (m) => { step = m.step; loss = m.loss; curve.push([m.step, m.loss]); });
      if (!playing) break;
    }
  }

  onDestroy(() => { playing = false; void engine?.dispose(); engine = null; });
</script>

<div use:inview={boot}>…</div>
```

Notes specific to Svelte 5:

- Runes only — `$state`, `$derived`, `$effect`, `$props`. No stores for this.
- `curve.push(...)` on a `$state` array is reactive; you do not need to reassign.
- For a model shared by several components, put the lab class in a
  `*.svelte.ts` module with `$state` fields and export a single instance. The
  page calls `disposeAll()` on unmount — GPU memory is no souvenir.
- `onDestroy` also runs after server prerender, so keep it browser-safe.

## React

```tsx
const engineRef = useRef<Engine | null>(null);
const [phase, setPhase] = useState<Phase>('idle');
const [metrics, setMetrics] = useState({ step: 0, loss: NaN });
const curve = useRef<Array<[number, number]>>([]);

useEffect(() => {
  let cancelled = false;
  (async () => {
    setPhase('loading');
    if (!(await detectWebGPU())) return setPhase('no-webgpu');
    const e = new Engine({ tokenData });
    await e.init(config);
    if (cancelled) { void e.dispose(); return; }   // StrictMode double-mounts
    engineRef.current = e;
    setPhase('ready');
  })();
  return () => { cancelled = true; void engineRef.current?.dispose(); engineRef.current = null; };
}, []);
```

React 18 StrictMode mounts effects twice in development. Without the `cancelled`
flag you get two workers competing for one GPU device and a boot that never
finishes. This is the single most common React + WebGPU bug.

Batch metric updates — `setMetrics` on every step will re-render hundreds of
times a second. Buffer into a ref and flush on rAF.

## Charts

SVG for loss curves, canvas for anything with thousands of points.

- Log-scale the y axis. Loss falls by orders of magnitude; a linear axis shows
  one big drop and then a flat line that hides all later progress.
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

Read colours from CSS custom properties **at draw time**, so the canvas follows
light/dark without a redraw hook:

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
- **No speed controls.** Pick one pace and tune it for reading.
- Status line shows live truth: `step 240 · loss 0.312 nats · 12 ms/step`, in
  tabular numerals so digits do not jitter.
- Disable, do not hide. A control that vanishes mid-run is disorienting.

## Say what the numbers mean

A loss of 0.35 means nothing on its own. Show the baseline: `uniform guess would
be 3.18 nats`. Label the units. When a knob has a hidden cost — changing the
learning rate resets Adam's moments, loading a checkpoint discards optimizer
history — say so in the caption rather than pretending it is free.

## Reset that actually resets

Keep the step-0 checkpoint from boot and reload it, rather than re-initialising:

```ts
this.initCkpt = await engine.exportCheckpoint();       // at boot
await engine.loadWeights(this.initCkpt.slice(0));      // on reset; slice keeps the master
```

Clear the curves and the samples at the same time. A reset that leaves the old
loss curve on screen is a bug report waiting to happen.
