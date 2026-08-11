# Ownership, references, and leaks

jax-js arrays are GPU buffers with **manual reference counting**. There is no
garbage collector for them. This page is the full model; SKILL.md laws 1 and 2
are its summary.

## The rules, exactly

1. A newly created array has `refCount === 1`.
2. **Passing an array into anything decrements it by one.** That includes
   `np.*` functions, methods on the array itself, `nn.*`, `random.*`, jitted
   functions, optax `update`/`applyUpdates`, and `tree.*` helpers.
3. `.ref` increments by one and returns the same array. Use it to lend an extra
   consumption.
4. When the count reaches zero the buffer is freed. Touching the handle after
   that throws `ReferenceError: Referenced tracer ... freed, please use .ref
   move semantics`.
5. `.dispose()` decrements by one explicitly. Disposing an already-free array
   throws.

Think of it as Rust's move semantics with an explicit `.clone()` spelled `.ref`.

```ts
const a = np.array([1, 2, 3]);   // rc 1
const b = a.ref;                 // rc 2, b === a
np.sum(b);                       // rc 1  (b consumed)
np.sum(a).item();                // rc 0  (a consumed, freed)
```

## What consumes, at a glance

| Call | Consumes its array arg(s)? |
| --- | --- |
| `np.sum(a)`, `np.dot(a, b)`, `np.square(a)`, `np.zerosLike(a)` … | **yes** |
| `a.add(b)`, `a.mul(2)`, `a.reshape(...)`, `a.transpose()`, `a.slice(...)` | **yes** (receiver *and* array operand) |
| `nn.relu(a)`, `nn.oneHot(ids, v)`, `nn.dotProductAttention(q, k, v)` | **yes** |
| `jitFn(a, b)` | **yes** |
| `solver.update(grads, state, params)`, `applyUpdates(params, updates)` | **yes**, all pytree args |
| `a.item()`, `a.js()`, `a.dataSync()`, `await a.data()` | **yes** — reading is a consumption |
| `await blockUntilReady(x)` | **no** |
| `a.ref`, `tree.ref(t)` | no — they *add* a reference |
| `a.shape`, `a.size`, `a.dtype`, `a.ndim`, `a.refCount` | no |

`.shape` and friends are the only free reads. Everything that moves data
consumes.

## The counting method

For each array-valued binding in a scope, count how many times it is used.
Write `.ref` on all uses but the last.

```ts
// x used 3 times → two .ref
const mean = np.mean(x.ref, -1, { keepdims: true });
const centred = x.ref.sub(mean.ref);
const variance = np.mean(np.square(x.sub(mean)), -1, { keepdims: true });
```

For a value used inside a loop, `.ref` it every iteration and dispose the master
copy once at the end:

```ts
const mask = np.array(buf).reshape([S, S]);
for (let h = 0; h < H; h++) scores[h] = raw[h].add(mask.ref);
mask.dispose();
```

Or, the pattern used throughout jaxverse: `.ref` on all but the known-last use,
tracked by an index test.

```ts
const lastUse = li === cfg.nLayer - 1 && h === H - 1;
const scores = qk.add(lastUse ? causalMask : causalMask.ref);
```

## Pytrees

A *pytree* is any nested plain object / array / tuple whose leaves are arrays.
Parameters, gradients and optimizer state are all pytrees.

```ts
import { tree } from '@jax-js/jax';

tree.ref(params)                       // +1 on every leaf
tree.dispose(params)                   // free every leaf
tree.map((x) => x.mul(2), params)      // consumes params, returns a new tree
tree.map((p, g) => p.sub(g.mul(lr)), params, grads)   // two trees, both consumed
tree.leaves(tree.ref(params))          // array of leaves (each holding a ref)
tree.flatten(t) / tree.unflatten(def, leaves)
```

Counting parameters — note the `tree.ref` and the explicit dispose, because
`tree.leaves` hands you live references:

```ts
export function paramCount(params: any): number {
  const leaves = tree.leaves(tree.ref(params)) as any[];
  const total = leaves.reduce((s, l) => s + l.size, 0);
  for (const l of leaves) l.dispose();
  return total;
}
```

## Per-step discipline in a training loop

Every array created inside the loop must be consumed or disposed inside it.

```ts
function trainStep(): number {
  const { tokenOH, posOH, targetOH } = makeBatch();          // 3 new arrays
  const [lossVal, grads] = jitStep(
    tree.ref(params), tokenOH.ref, posOH.ref, targetOH.ref); // lend to jit
  const [updates, nextState] = solver.update(grads, optState, tree.ref(params));
  params = applyUpdates(params, updates);
  optState = nextState;
  tokenOH.dispose(); posOH.dispose(); targetOH.dispose();    // our copies
  return lossVal.item();                                     // consumes lossVal
}
```

Why `.ref` on the batch tensors *and* an explicit `dispose`? Because the jitted
function consumes what it is given. Lending a ref keeps a handle so the batch can
be reused (for a metric, say) and then freed deterministically. If a batch tensor
is used exactly once, drop the `.ref` and the `dispose` together.

## Freeing the model

```ts
export function disposeTree(t: any): void {
  for (const leaf of tree.leaves(t)) leaf.dispose();
}

// teardown
disposeTree(params);
disposeTree(optState);
jitStep.dispose();      // jit returns an OwnedFunction holding constants
jitForward.dispose();
```

In a worker, also terminate the worker — a worker that never releases its
`GPUDevice` blocks the next one from acquiring it:

```ts
async dispose() {
  try {
    // a worker mid-jit answers nothing; give the graceful path a deadline
    await Promise.race([this.call('dispose'), new Promise((r) => setTimeout(r, 400))]);
  } finally {
    this.worker.terminate();
    this.pending.clear();
  }
}
```

## Finding a leak

Symptoms: steps get slower, then the tab dies, or WebGPU reports out of memory.

1. Search every per-step code path for an array created and not consumed.
   Batch tensors, masks, and position ids are the usual suspects.
2. Check both branches of every conditional read:
   `if (n % 50 === 0) log(l.item()); else l.dispose();`
3. Check error paths — a `throw` between create and dispose leaks.
4. `a.refCount` in a debugger tells you what is still held.
5. `import { profiler } from '@jax-js/jax'; profiler.startTrace()` shows kernel
   activity and gaps.

## Finding a double-free

Symptom: `Referenced tracer Array:float32[...] freed, please use .ref move
semantics`, thrown from a line that *looks* fine.

The throw happens at the **second** use — the bug is at the first. Work
backwards: what else consumed this array earlier? Nine times in ten it is one of

- `x.item()` followed by `x.dispose()`
- a value used twice in one expression: `np.sum(x.mul(x))` → `np.sum(x.ref.mul(x))`
- a pytree passed to `valueAndGrad` and then to `solver.update` without
  `tree.ref`
- a module-level constant consumed inside a loop without `.ref`
