# API surface

Condensed from the `@jax-js/jax@0.1.21` type definitions. Everything consumes
its array arguments — see [memory.md](memory.md).

## Top-level exports

```ts
import {
  init, defaultDevice, devices, devicePut, getWebGPUDevice, blockUntilReady,
  numpy as np, nn, random, lax, scipySpecial, tree,
  jit, grad, valueAndGrad, vmap, jvp, vjp, linearize,
  jacrev, jacfwd, jacobian, hessian, makeJaxpr,
  profiler, setDebug,
  Array, DType, type Device, type JsTree, type OwnedFunction,
} from '@jax-js/jax';
```

`Device = 'cpu' | 'wasm' | 'webgpu' | 'webgl'`.

## Creating arrays

```ts
np.array([1, 2, 3])                              // from JS array
np.array(new Float32Array(n)).reshape([r, c])    // from a typed array — the fast path
np.array(buf, { dtype: np.int32 })
np.zeros([2, 3]) / np.ones(...) / np.full(shape, v) / np.zerosLike(a) / np.onesLike(a)
np.arange(10) / np.linspace(0, 1, 50) / np.eye(4) / np.tri(n)
```

dtypes: `np.float32` (default), `np.float64`, `np.float16` (WebGPU only),
`np.int32`, `np.uint32`, `np.bool`. No complex, no bfloat16. `int8`/`uint8` are
emulated.

Build device arrays from **typed arrays**, not nested JS arrays — one copy
instead of a parse.

## Array properties and readback

```ts
a.shape  a.size  a.dtype  a.ndim  a.device  a.refCount   // free, non-consuming
a.item()            // scalar → number        CONSUMES
a.js()              // nested JS array        CONSUMES
a.dataSync()        // TypedArray, blocking   CONSUMES
await a.data()      // TypedArray, async      CONSUMES
await blockUntilReady(treeOrArray)             // does not consume
await a.gpuBuffer() / a.gpuBufferSync()        // share with your own WebGPU code
```

Prefer `await a.data()` over `dataSync()` on the main thread; inside a worker
`dataSync()` is fine and simpler.

## Fluent methods (on every array/tracer)

```
neg add sub mul div mod
greater less equal notEqual greaterEqual lessEqual
sum prod mean min max all any        (axis?, { keepdims })
transpose(perm?) reshape(shape) astype(dtype) view(dtype)
flatten ravel sort argsort diagonal slice(...)
```

Scalars work directly: `x.mul(2)`, `x.add(1e-5)`.

`slice` takes one entry per axis: a number, `[start, stop]`, `[start]`, or `[]`
for "all".

```ts
a.slice([1, 3])          // rows 1..2, all columns
a.slice([], [0, 2])      // all rows, columns 0..1
a.slice(2)               // row 2, rank reduced
```

## numpy module

Arithmetic and math: `add subtract multiply divide power sqrt square exp log
log1p log2 log10 sin cos tan tanh sinh cosh sign abs floor ceil round clip
maximum minimum where sign reciprocal`

Reductions: `sum prod mean std var_ min max argmin argmax cumsum cumprod ptp
average median`-adjacent helpers.

Shape: `reshape transpose swapaxes moveaxis expandDims squeeze concatenate stack
hstack vstack dstack columnStack split tile repeat pad flip roll broadcastTo
meshgrid take takeAlongAxis`

Linear algebra: `dot matmul matvec vecmat inner outer tensordot einsum trace`,
plus `np.linalg.{inv, solve, det, slogdet, cholesky, svd, eigh, lstsq,
matrixPower, vectorNorm, matrixNorm}`. **No QR, no `eig`, no `pinv`.**

`np.fft.*` is complete and fast.

## nn module

```ts
nn.relu relu6 leakyRelu elu celu selu gelu silu/swish mish softplus squareplus
   sigmoid hardSigmoid hardSilu hardTanh logSigmoid softSign sparsePlus glu
nn.softmax(x, axis)     nn.logSoftmax(x, axis)
nn.logsumexp(x, axis)   nn.logmeanexp   nn.standardize
nn.oneHot(ids, numClasses)
nn.dotProductAttention(q, k, v, { isCausal?, mask?, scale?, bias? })
```

`dotProductAttention` takes `[B, S, H, headDim]` (batch, sequence, heads, dim)
and returns the same shape. It never materialises the attention matrix — if you
want to *see* the weights, compute them by hand (see
[models.md](models.md#attention-you-can-look-at)).

## random

Explicit keys, like JAX. Threefry2x32, bitwise-identical to JAX's output.

```ts
const key = random.key(42);
const keys = random.split(key, n);        // [n, 2] — slice one row per draw
random.normal(k, [rows, cols])
random.uniform(k, shape, { minval, maxval })
random.randint(k, shape, low, high)
random.categorical(k, logits, { axis })   // on-device sampling
random.bernoulli / choice / permutation / gumbel / exponential / …
```

Keys are arrays and obey the ownership rules. A common helper:

```ts
const keys = random.split(random.key(seed), n);
let ki = 0;
const nk = () => { ki++; return ki < n ? keys.ref.slice(ki - 1) : keys.slice(ki - 1); };
```

For sampling loops that need a plain JS RNG (topk/temperature sampling on the
host), use a seeded PRNG such as mulberry32 — do not burn device keys per token.

## Transformations

```ts
jit(f, { staticArgnums?: number[], device?: Device })   // → OwnedFunction
grad(f, { argnums?, hasAux? })
valueAndGrad(f, { argnums?, hasAux? })                  // → [value, grads]
vmap(f, inAxes?)
vjp(f, primals, { hasAux? })                            // → [out, vjpFn]
jvp(f, primals, tangents)
jacrev / jacfwd / jacobian / hessian / linearize
lax.stopGradient(x)                                     // detach — PPO/DPO ratios
```

`argnums` selects which argument to differentiate (default 0); an array returns a
tuple of gradients.

```ts
const [loss, [gx, gy]] = valueAndGrad(f, { argnums: [0, 1] })(x.ref, y.ref);
const [[loss, aux], grads] = valueAndGrad(f, { hasAux: true })(params);
```

`hasAux` is how you return metrics (accuracy, per-example loss) from the same
traced computation without a second pass.

## optax

```ts
import {
  sgd, adam, adamw, chain, applyUpdates,
  clipByGlobalNorm, addDecayedWeights, scale, scaleBySchedule, scaleByLearningRate,
  squaredError, l2Loss, treeNorm, treeSum, treeZerosLike,
} from '@jax-js/optax';

const solver = chain(clipByGlobalNorm(1.0), adamw(3e-4, { weightDecay: 0.01 }));
let optState = solver.init(tree.ref(params));

const [updates, nextState] = solver.update(grads, optState, tree.ref(params));
params = applyUpdates(params, updates);
```

A learning-rate *schedule* is a plain `(count: number) => number`:

```ts
const warmupCosine = (peak: number, warmup: number, total: number) => (t: number) =>
  t < warmup ? (peak * t) / warmup
             : peak * 0.5 * (1 + Math.cos((Math.PI * (t - warmup)) / (total - warmup)));
const solver = adam(warmupCosine(3e-4, 100, 5000));
```

**optax cannot run inside `jit`** at 0.1.2 — `treeBiasCorrection` calls
`count.item()`. To fuse the optimizer into the compiled step, hand-roll it:
`templates/fused-adam.ts`.

Changing the learning rate mid-run means rebuilding the solver, which resets Adam
moments. That is an honest cost — say so in the UI rather than hiding it.

## Not supported

`lax.scan`, `lax.while_loop`, `lax.cond`, `custom_vjp`/`custom_jvp`,
`checkpoint`/rematerialisation, `pmap`/`shard_map`, sparse arrays, complex
dtypes, QR/eig/pinv, `jax.scipy` beyond `scipy.special`. Loops are ordinary JS
loops — they unroll into the trace, so keep them short and shape-static.

Also: **`grad` through `np.take` throws inside `jit`.** Use one-hot matmuls for
anything differentiated (SKILL.md law 4).

## Debugging

```ts
import { setDebug, profiler, makeJaxpr } from '@jax-js/jax';
setDebug(1);                                  // log kernel compilation
console.log(makeJaxpr(f)(x).jaxpr.toString()); // inspect the traced graph
profiler.startTrace();                        // kernel timeline
```
