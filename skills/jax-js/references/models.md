# Model recipes

Working code lives in `templates/`; this page is the reasoning behind it and the
parts you have to decide yourself.

## Shape discipline

Fold the batch into rows before every projection. A `[B, S, D]` tensor becomes
`[B·S, D]`, every weight is a plain 2-D matrix, and every layer is one matmul.

```ts
let x = np.dot(tokenOH.reshape([-1, vocab]), params.wte);   // [B·S, D]
```

Reshape back to `[B, S, H, headDim]` only where an op demands it
(`nn.dotProductAttention`), then flatten again. Keeping one rank-2 convention
across the model makes the ownership bookkeeping tractable and keeps jit
signatures stable.

## Initialisation — the failure that looks like a learning-rate bug

**Never zero-initialise an output projection.** Zero-init on `wo`, `mlpFc2` or
the classifier head blocks all gradient into the block interior at step 0: the
loss sits flat, you lower the learning rate, and nothing changes.

Use small-random instead — enough signal to start, small enough not to blow up
the residual stream:

| Tensor | Init |
| --- | --- |
| token / positional embeddings | `normal × 0.02` |
| Q, K, V projections | uniform ±√(3/D) |
| attention output `wo` | uniform ±0.2·√(3/D) |
| MLP in `fc1` | uniform ±0.4·√(3/D) |
| MLP out `fc2` | uniform ±0.2·√(3/D) |
| LM head | `normal × 0.001` |
| MLP hidden layers | Glorot: uniform ±√(6/(fan_in+fan_out)) |
| biases | zeros (safe — they are not a gradient path) |

## MLP — `templates/model-mlp.ts`

One configurable MLP covers curve fitting, 2-D classification, MNIST,
autoencoders and VAEs. Its config is the whole design surface:

```ts
{ layers: [2, 16, 16, 2], activation: 'tanh', loss: 'xent', seed: 7 }
```

- `loss: 'mse'` expects a float target matrix; `'xent'` expects a **one-hot**
  `[n, classes]` matrix (build it outside the jitted step).
- An autoencoder is just `y = x` with `'mse'` and a narrow middle layer.
- `vae: { at, beta }` turns layer `at` into a Gaussian bottleneck: it emits
  `2 × width` numbers (a mean and a log-variance), and the tail receives a
  reparameterised sample while training or the mean on every read-out path.
  Because the loss averages over elements, the textbook β = 1 is `1 / outDim`.
  Keep the waist linear — a non-linearity there fights the prior.

The KL term, written out:

```ts
// KL(N(μ, σ²) ‖ N(0, I)) = ½ Σ (μ² + σ² − 1 − log σ²)
const kl = np.mean(np.sum(np.square(mu).add(np.exp(logvar.ref)).sub(1).sub(logvar), -1)).mul(0.5);
loss = loss.add(kl.mul(beta));
```

Pass the noise in as an argument (`random.normal(key, [n, latent])` built
outside the step) rather than drawing it inside — the jitted function then has
no hidden state and stays reproducible.

## Transformer — `templates/model-transformer.ts`

A decoder-only stack: pre-norm RMSNorm, causal attention, 4× MLP, no biases, no
dropout. Weight-tying is *not* used — a separate `lmHead` trains faster at these
sizes.

Sizes that train at interactive speed on a laptop GPU:

| Purpose | nLayer | nEmbd | nHead | blockSize | vocab | params |
| --- | --- | --- | --- | --- | --- | --- |
| toy / unit test | 2 | 32 | 4 | 16 | 8 | ~26k |
| character LM, live demo | 2 | 96 | 4 | 96 | 24–100 | ~235k |
| word-piece LM | 4 | 128 | 4 | 128 | 1–4k | ~1.3M |
| the practical ceiling | 6 | 256 | 8 | 256 | 4k | ~7M |

Rules of thumb: `nEmbd % nHead === 0`; keep `vocab × blockSize × batch` modest,
because the one-hot batch is `B·S·V` floats; and start from
`lr ≈ 1.5e-3` with Adam `b2 = 0.99` for models under a million parameters.

### Padding beats recompiling

Generation grows the prompt one token at a time. Tracing a new kernel for every
length is catastrophic. Instead right-pad to `blockSize` and always call the
same signature — causal attention makes positions after the last real token
irrelevant to the rows you read:

```ts
const buf = new Int32Array(S);                       // zero-filled
for (let i = 0; i < Math.min(tokens.length, S); i++) buf[i] = tokens[i];
const lp = jitForward(oneHot(buf), oneHot(positions));   // [S, V]
const row = lp.subarray((tokens.length - 1) * V, tokens.length * V);   // the only row you want
```

One jit signature then serves every prompt length.

### Sampling in lockstep

To draw G continuations of one prompt (RLVR groups, beam-ish exploration, "show
me five futures"), advance all G rows together in one `[G, S]` forward per token
rather than G separate loops. See `handleSampleGroup` in
[rl.md](rl.md#group-sampling).

### Attention you can look at

`nn.dotProductAttention` never materialises the attention matrix, which is what
makes it fast. To *show* attention, compute it by hand with per-head 2-D
matmuls — and keep the numerics identical to the fused path, or the picture is a
lie about the model you trained:

```ts
const scores = np.dot(qh, kh.transpose()).mul(1 / Math.sqrt(headDim)).add(causalMask.ref);
const w = nn.softmax(scores, -1);          // [S, S] — this is the picture
outs.push(np.dot(w.ref, vh));
```

`templates/model-transformer.ts` ships `forwardWithAttention`, and
the skill repo's [tests/api.test.mjs](https://github.com/NeoVand/jax-js-skill/blob/main/tests/api.test.mjs) asserts it agrees
with the fused forward to 2e-3. If you
edit one, edit both.

### Reading the residual stream

`forwardResiduals` returns the activation after every block, `[S, nEmbd]` per
layer. That is the surface a linear probe reads to ask whether the model has
built an internal representation of something (the Othello-GPT experiment).
Train a small linear classifier on those vectors against the property you care
about.

## Convolutions

`lax.conv`, `lax.convGeneralDilated` and `lax.reduceWindow` exist and are
differentiable, but there is no `nn.Conv2d` layer object — you carry the weights
yourself, as with everything else. For MNIST-scale demos an MLP on flattened
pixels is usually the better trade: fewer moving parts, and it trains in seconds.
Reach for convolutions when the demo is *about* convolution.

## Loss functions

```ts
// mean squared error
np.mean(np.square(pred.sub(target)))

// cross-entropy from one-hot targets (no gather needed)
const logp = nn.logSoftmax(logits, -1);
np.mean(np.sum(logp.mul(targetOH), -1).neg())

// next-token NLL over a whole batch of sequences
np.mean(np.sum(logprobs.mul(targetOH.reshape([-1, vocab])), -1).neg())
```

Report language-model loss in **nats per token**, and say what the uniform
baseline is (`Math.log(vocab)`) — otherwise the number means nothing to a
reader. If you compare two tokenizers, convert to **bits per character**:
`nats / Math.LN2 / charsPerToken`.

## Honest validation

Hold out a tail of the data and evaluate on **fixed** batches drawn with a
constant seed, so the curve is comparable across time:

```ts
valStart = Math.floor(data.length * 0.95);
function valLoss() {
  const vr = mulberry32(9999);          // same batches every call
  let total = 0;
  for (let i = 0; i < 4; i++) {
    const b = makeBatchOH(cfg, data, vr, BATCH, valStart, data.length);
    total += jitLoss(tree.ref(params), b.tokenOH, b.posOH, b.targetOH).item();
  }
  return total / 4;
}
```

## Checkpoints

Flatten the pytree to one `Float32Array` in `tree.leaves` order and rebuild
against a freshly-initialised template of the same config. That gives you: the
twin-worker courier, a reset-to-step-0 button, a time-machine scrubber, and
model export — all from twenty lines. See `flattenParams` / `loadParams` in
`templates/model-transformer.ts`.

For storage, quantise to int8 with a per-tensor scale; a 5M-parameter model
becomes 5 MB instead of 20 MB, and sampling quality is visually unchanged at
these sizes.

## Fine-tuning is the same machine with different data

There is no separate code path. Swap the corpus in place and keep training:

```ts
await engine.setTokens(curatedTokens);   // params, optimizer state and jits survive
await engine.train(200, onMetrics);      // this IS supervised fine-tuning
```

Lower the learning rate when you do (3e-4 rather than 1.2e-3) — fine-tuning
that moves as fast as pretraining just overwrites what pretraining learned.
