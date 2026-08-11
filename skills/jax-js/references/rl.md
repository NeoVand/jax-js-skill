# Reinforcement learning and preference training

Everything here is ordinary supervised machinery with a different weight on each
token. There is no special RL API; you build the loss.

## The one idea

Policy gradient is cross-entropy with a per-token coefficient. Where supervised
training says "make this token more likely", REINFORCE says "make this token
more likely *in proportion to how good the outcome was*".

```
supervised:  L = −Σ log π(token)
REINFORCE:   L = −Σ Â · log π(token)
```

So the same jitted forward, the same one-hot targets, and one extra `[B·S, 1]`
weight tensor multiplied in. That is the whole implementation.

## REINFORCE / RLVR step

`seqs` is `[G, blockSize+1]` Int32 with `-1` padding; `advs[g]` is the advantage
for sequence `g`; `starts[g]` is the index of the first **generated** token —
prompt positions must get no gradient, or the model trains on its own prompt.

```ts
function rlStep(seqs: Int32Array, advs: number[], starts: number[]) {
  const S = cfg.blockSize, G = advs.length;
  const inputBuf = new Int32Array(G * S);
  const targetBuf = new Int32Array(G * S);
  const wBuf = new Float32Array(G * S);
  let credited = 0;

  for (let g = 0; g < G; g++) {
    for (let t = 0; t < S; t++) {
      const a = seqs[g * (S + 1) + t];
      const b = seqs[g * (S + 1) + t + 1];
      inputBuf[g * S + t] = a < 0 ? 0 : a;
      targetBuf[g * S + t] = b < 0 ? 0 : b;
      // position t predicts row t+1 — credit only generated positions
      if (a >= 0 && b >= 0 && t + 1 >= starts[g]) { wBuf[g * S + t] = advs[g]; credited++; }
    }
  }
  if (credited === 0) return { loss: 0, skipped: true };
  for (let i = 0; i < wBuf.length; i++) wBuf[i] /= credited;   // normalise by token count

  const tokenOH  = nn.oneHot(np.array(inputBuf,  { dtype: np.int32 }).reshape([G, S]), cfg.vocab);
  const posOH    = nn.oneHot(np.tile(np.arange(S).astype(np.int32), [G, 1]), cfg.blockSize);
  const targetOH = nn.oneHot(np.array(targetBuf, { dtype: np.int32 }).reshape([G, S]), cfg.vocab);
  const w        = np.array(wBuf).reshape([G * S, 1]);

  jitRl ??= jit((p, tok, pos, tgt, ww) =>
    valueAndGrad((pp) => {
      const logp = forwardLogprobs(pp, cfg, S, tok, pos);       // [G·S, V]
      return np.sum(logp.mul(tgt.reshape([-1, cfg.vocab]).mul(ww))).neg();
    })(p));

  const [lossVal, grads] = jitRl(tree.ref(params), tokenOH.ref, posOH.ref, targetOH.ref, w.ref);
  const [updates, next] = solver.update(grads, optState, tree.ref(params));
  params = applyUpdates(params, updates);
  optState = next;
  tokenOH.dispose(); posOH.dispose(); targetOH.dispose(); w.dispose();
  return { loss: lossVal.item() };
}
```

Normalising by the number of credited tokens (rather than by `G`) keeps the
gradient scale independent of how long the generations happen to be.

## GRPO — group-relative advantages

Sample G continuations of the *same* prompt, score them, and standardise within
the group. No value network, no critic:

```ts
const rewards = seqs.map(verify);                       // your verifier
const mean = rewards.reduce((a, b) => a + b, 0) / G;
const sd = Math.sqrt(rewards.reduce((a, r) => a + (r - mean) ** 2, 0) / G) || 1;
const advs = rewards.map((r) => (r - mean) / sd);       // group-relative
```

If every sample in a group gets the same reward the advantages are all zero and
the step is a no-op — that is correct, not a bug. Skip it and resample.

Compute advantages on the host in plain JavaScript. There is nothing to gain
from putting a G-element standardisation on the GPU.

## Group sampling

Advance all G rows in lockstep — one `[G, S]` forward per generated token
instead of G independent loops. This is the difference between RLVR being
interactive and being unusable.

```ts
const seqs = Array.from({ length: G }, () => prompt.slice());
const done = Array(G).fill(false);
for (let t = 0; t < maxNew; t++) {
  const L = prompt.length + t;                     // lockstep: every live row has length L
  const buf = new Int32Array(G * S);
  for (let g = 0; g < G; g++)
    for (let i = 0; i < Math.min(seqs[g].length, S); i++) buf[g * S + i] = seqs[g][i];

  const lp = jitForwardBatch(tree.ref(params),
    nn.oneHot(np.array(buf, { dtype: np.int32 }).reshape([G, S]), cfg.vocab),
    nn.oneHot(np.tile(np.arange(S).astype(np.int32), [G, 1]), cfg.blockSize));
  const data = lp.dataSync() as Float32Array;      // [G·S, V]

  let allDone = true;
  for (let g = 0; g < G; g++) {
    if (done[g]) continue;
    const row = data.subarray((g * S + L - 1) * cfg.vocab, (g * S + L) * cfg.vocab);
    const next = sampleFromRow(row, temperature, topK, rng);
    if (next === stopToken) { done[g] = true; continue; }
    seqs[g].push(next);
    allDone = false;
  }
  if (allDone) break;
}
```

## Learning rates for RL

Policy gradient without a KL leash needs a **much** smaller step than
pretraining. At pretraining rates the policy collapses into repeated tokens
within a dozen updates — this is measured, not theoretical.

| Stage | Typical lr |
| --- | --- |
| pretraining | 1.2e-3 |
| supervised fine-tuning | 3e-4 |
| REINFORCE / GRPO, no KL term | 1e-4 |

## The KL leash

Plain REINFORCE will happily wander to a degenerate policy that scores well and
writes nothing. Keep a frozen copy of the reference policy and add
`β · KL(π ‖ π_ref)` to the loss.

Cheapest honest version: keep `π_ref` in a *second worker* (the reference model
never trains, so it can hold a frozen checkpoint), get its log-probs for the
sampled sequences, and add the per-token difference as an extra weighted term.
The twin-worker machinery in [workers.md](workers.md) is exactly the right
shape for this — one worker trains, the other answers.

Without a KL term, say so. A demo that calls itself RLHF and has no reference
model is misleading the reader.

## DPO

Direct preference optimisation needs no sampling loop and no verifier — just
pairs `(chosen, rejected)` and the frozen reference log-probs:

```
L = −log σ( β · [ (logπ(chosen) − logπ_ref(chosen)) − (logπ(rejected) − logπ_ref(rejected)) ] )
```

Implementation notes:

- Sum log-probs over the **completion** tokens only, exactly as in `rlStep`.
- The reference terms are constants: compute them once per pair with the frozen
  model and pass them in as a `[B]` array. Do not recompute them under the
  gradient — and if you keep the reference model live, wrap its output in
  `lax.stopGradient`.
- `β` is typically 0.1–0.5. Both terms are already log-probs; do not add a
  second softmax.

## Bandits and classic control

For tabular bandits, gridworlds and cart-pole, jax-js is usually the wrong tool:
the arithmetic is a handful of floats per step and plain JavaScript is faster and
far easier to read. Use jax-js when the policy is a *network* whose gradient you
need. Mixing both in one page is fine — a policy-gradient gridworld with a
2-layer MLP policy is a good use, a Q-table is not.

## Reward hacking is a feature of the demo

If the environment can be gamed, a policy-gradient loop will find it, quickly and
visibly. That is worth showing rather than patching: let the reader watch the
verifier's score climb while the behaviour gets worse, then add the leash and
watch it hold.
