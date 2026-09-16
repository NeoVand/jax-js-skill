# Latent world models and learned control

Use this reference for action-conditioned prediction, JEPA-style learning and
planning demos. These are lessons from the browser world-model chapter, not
performance claims about every architecture or a recipe for reproducing a paper.

## State the experiment before choosing the network

Write down what the model receives, predicts and optimizes. A useful contract is:

- observation `o[t]`: the actual image or sensor measurement;
- action `a[t]`: applied during the transition from `t` to `t+1`;
- encoder `z[t] = E(o[t])` and predictor `zHat[t+1] = P(history, actions)`;
- predictor training: compare predicted and encoded future observations;
- planner: search candidate actions with frozen model weights;
- simulator: supplies training observations and measures executed behavior.

Keep simulator state labels out of the encoder/predictor loss when claiming
learning from pixels alone. A separately fitted display readout may use such
labels; explain that supervision and keep it out of planning. Self-supervised
dynamics can later support a goal-specific planner without retraining the model.

A single image may not reveal velocity. Give the predictor enough observation
history, or demonstrate that ambiguity explicitly. Align previous actions with
the transitions they caused; an off-by-one action bug can still produce a
plausibly falling loss on a smooth system.

## Respect the chosen objective

[LeWorldModel v3](https://arxiv.org/abs/2603.19312v3) jointly trains an encoder
and action-conditioned predictor with embedding prediction and SIGReg losses.
Unlike JEPA variants with detached/EMA targets, LeWM propagates gradients through
both branches. Do not import stop-gradient from a different method. In JaxJS,
`.ref` only manages ownership; `lax.stopGradient` changes the derivative.

A constant encoder can make prediction error small without encoding useful
information. Show both loss terms, representation spread and held-out task
evidence. A non-collapsed representation alone does not establish useful dynamics.

When porting a distributional regularizer, verify the reference implementation's
axes, scaling and randomness. For a tensor `[batch, time, features]`, do not
silently flatten batch and time into independent samples if the method computes
statistics across examples at each time position. Microbatch gradient
accumulation reproduces a separable mean loss (with correct weighting), but
averaging independently computed batch statistics generally changes a
batch-coupled objective. Recheck regularization strength after changing batches.

Use a tiny deterministic numerical fixture to compare the ported loss and
selected gradients. Then run a collapse ablation with matched initialization,
data, update count and evaluation examples. Avoid making the baseline deliberately
weak through different scaling, training time or data.

## Evaluate learning, rollout and planning separately

| Question | Evidence |
| --- | --- |
| Does the encoder preserve useful distinctions? | Spread plus held-out discrimination or probe quality |
| Does the predictor improve on persistence? | Fixed held-out one-step futures against copying the current embedding |
| Does the model use the action? | Same history, different proposed actions; compare with observed outcomes |
| Does prediction remain useful over time? | Open-loop rollout error by horizon, without feeding true future frames back in |
| Does the planner help? | Executed goal error/success against a simple policy on fixed starts/goals |

For retrieval, label chance as `1 / candidateCount`, keep candidate difficulty
fixed, and define tie handling. Ties must not favor the correct candidate.
Raw distances from two independently learned embedding spaces are not directly
comparable; use normalized or task-level evidence and report its definition.

Freeze the checkpoint while comparing before/after or counterfactual futures.
Tag asynchronous evaluation with model revision, checkpoint step, data seed and
horizon; reject results from a previous reset. Start with deterministic CPU/wasm
fixtures, then test training and inference on the actual browser backend.

## Make the visual evidence faithful

Draw an architecture diagram that matches the implementation: observations,
shared encoder, history/actions, predictor, future target and each loss. Mark
gradient paths and distinguish training from planning. Label an MLP adaptation
as such rather than drawing the paper's transformer while running an MLP.

Overlay learned rollout and simulator replay from **the same initial state and
action sequence**, advancing on the same observation clock. If the setup starts
at equilibrium, release may correctly do nothing; use an explicit shared warm-up
history when the lesson needs visible momentum. Do not add hidden motion to the
replay merely to make a prediction appear correct.

A decoded ghost mixes prediction error with readout error. Measure the readout
on true held-out embeddings separately. A renderer that enforces valid arm
geometry cannot by itself prove the representation learned those constraints.
Keep the physics replay independent of learned predictions.

Check sensor pixels against the visible scene using asymmetric poses: coordinate
signs, origin, crop and orientation should agree. Enlarge the preview with CSS
without claiming higher model resolution. Dark-mode inversion belongs in the
display layer; keep model input normalization stable across themes.

## Budget the browser experiment

A 32×32 to 64×64 change quadruples pixels. For a flattened grayscale MLP encoder
with first width `H`, its first weight matrix grows from `1024H` to `4096H`
parameters; gradients and Adam moments multiply that cost. Dataset images,
activations and upload traffic also grow. Preview dimensions are a separate knob.

Choose MLP, convolution or attention by the needed history/spatial structure and
measured held-out quality per second. A short fixed-history MLP can be effective
for a tiny world; a transformer can provide useful variable-context structure.
Neither is a universal winner. Compare compile time, steady-state training,
inference latency and memory at matched evaluation quality.

For the teaching interface, keep the loss visible alongside one readable
learning test. Show comparison/evaluation progress as it happens, label stale
results by step, and let the reader change a meaningful cause and see its effect.
See [ui.md](ui.md) for the general design checks.
