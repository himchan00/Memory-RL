# Symbolic Alchemy: what was tried, why, and what came back

Status as of 2026-09-03. Companion to `alchemy_status.md`, which describes the
environment; this file is the experiment ledger. Every number is a final value
at 24,064 rollouts (94 evals) unless the run is marked short.

---

## 0. How to read the two scoreboards

The project has two metrics that measure unrelated things, and most of the
confusion in this log comes from conflating them.

**`eval/return` — did the agent earn anything.** One episode is 10 trials x 20
steps = 200 steps. Each trial refreshes 3 stones and 12 potions; the hidden
chemistry is fixed for the whole episode. Reward comes *only* from dropping a
stone in the cauldron, and equals the stone's latent coordinate sum, with a +12
bonus at the best vertex:

| latent coords | reward |
|---|---|
| (-1,-1,-1) | -3 |
| one +1 | -1 |
| two +1 | +1 |
| (+1,+1,+1) | **+15** |

Using a potion pays nothing. Potions only flip one coordinate of one stone.
Return is the sum over all 10 trials, so at most 30 stones.

| reference | return |
|---|---|
| uniform random | 17.1 |
| `random_stone_potion` — **no-chemistry floor** | **145.2** |
| `chemistry_oracle` — **planner ceiling** | **287.1** |

`normalized = (return - 145.2) / 141.9`.

The floor is high because the observation **leaks each stone's true latent
reward** (`perceived_stone.reward` passes unchanged through `LatentStone ->
AlignedStone -> PerceivedStone`). "Cash the stones that are already positive,
discard the rest" scores 145 while knowing nothing. Chemistry's only value is
planning *which potion turns a -1 stone into a +15 stone*.

**`train/aux_canon_potion_acc` — did the agent figure the chemistry out.** A
separate exam, with no effect on reward. Each of the 12 potion slots holds one
of 6 latent types (3 axes x 2 directions), presented under a per-episode random
disguise. The auxiliary head emits 6 logits per slot; a slot scores 1 if the
argmax matches. Absent and used slots are excluded; averaged over present slots
and all timesteps (`policies/models/policy_rnn_dqn.py:792`).

| reference | accuracy |
|---|---|
| chance | 0.1667 |
| memoryless ceiling (`scripts/probe_frame_map.py`) | 0.1675 |
| MATE with aux loss, detached site | 0.553 |

The memoryless ceiling being at chance is the whole point: this quantity is
*only* obtainable from memory.

---

## 1. The original experiment and the thing that is failing

The intended study is a 2-axis grid: **algorithm** (DDQN, V-MPO, IMPALA, PPO)
x **memory architecture** (MATE, Transformer-XL, LSTM, ground-truth chemistry,
Markov). Ground-truth chemistry is the ceiling of the *memory* axis, not an
algorithm.

Before that grid means anything, the memory axis has to separate from the floor.
It does not. Everything sits in a band around the no-chemistry floor:

| run | memory | return | normalized |
|---|---|---|---|
| `mem_mate_sp_s42` | MATE | **150.4** | +0.04 |
| `mem_gpt_sp_s42` (14/94, OOM) | GPT-2 | 124.4 | -0.15 |
| `mem_lstm_sp_s42` (17/94, OOM) | LSTM | 108.0 | -0.26 |
| `orc_sp_base` | ground-truth chemistry, perceived frame | 156.5 | +0.08 |

The last row is the alarming one. **An agent handed the true chemistry as an
input scores 156.5** — six points above an agent with no chemistry at all. So
the failure is not "MATE cannot remember." Something upstream prevents *any*
agent from converting chemistry into reward, and until that is fixed the memory
comparison measures nothing.

---

## 2. Raising the oracle (which part: the observation / the RL horizon)

If the oracle cannot use chemistry it is handed, find out what is blocking it.
All of these hold the memory axis fixed at "true chemistry in the observation."

| # | intervention | what it touched | return | delta |
|---|---|---|---|---|
| 2.1 | `oracle_gamma99_control` | baseline | 158.4 | — |
| 2.2 | `oracle_gamma999` (discount 0.99 -> 0.999) | RL horizon | 158.1 | **-0.3** |
| 2.3 | `oracle_canon_s42` (`canonicalize_oracle`) | observation frame | 193.5 | **+35** |
| 2.4 | `oracle_potions` (`structured_potions`) | observation encoding | 225.3 | **+32** |
| 2.5 | `oracle_full_p0` (+ `context_graph_only`) | observation content | **232.3** | **+7** |
| 2.6 | `oracle_full_w256` (width 256) | network capacity | 235.7 | +3.4 |
| 2.7 | `canon_lr1e4` (critic lr 3e-5 -> 1e-4) | optimization | 211.5 | -21 |

**Reading.** The credit-assignment horizon is not the problem (2.2 moves
nothing across a 100x change in effective horizon). Capacity is not the problem
(2.6 gives +3.4 for 4x the width). Optimization is not the problem (2.7 is
worse). **Every real gain came from making the chemistry easier to *read*, not
from making the agent stronger.** `canonicalize_oracle` rewrites the
observation into the latent frame so the network no longer has to invert the
per-episode rotation and permutation itself; that single change is +35, and the
frame-related levers together take 158 -> 232 (normalized 0.09 -> 0.61).

The diagnosis: the blocker is the **perceived -> latent frame map**. The agent
is given every fact it needs to compute it and never does.

`scripts/probe_frame_map.py` confirms the map is easy in isolation — it is a
deterministic function of (perceived obs, `chem_gt[12:28]`), and the very same
critic MLP fits it to 100% test accuracy inside one epoch. The scalar TD signal
simply never drives the network to compute it.

---

## 3. Can the oracle be raised without privilege? (No.)

2.3 and 2.5 are privileged: they hand the agent the latent frame. A memory model
cannot use them. So: can the same 158 -> 232 be reached with fair
re-encodings only? Nine attempts, all with `canonicalize_oracle=False`:

| run | lever | return |
|---|---|---|
| `orc_sp_base` | baseline | **156.5** |
| `orc_sp_ss` | `structured_stones` | 156.4 |
| `orc_sp_fact` | factored potions | 154.7 |
| `orc_sp_phase` | `add_trial_phase` | 152.8 |
| `orc_sp_slots` | slot re-encoding | 149.8 |
| `orc_sp_all` | all of the above | 150.4 |
| `orc_sp_big` | wider net | 148.2 |
| `orc_ctx_film` | route context via FiLM | 150.2 |
| `orc_ctx_hyper` | route context via hypernetwork | 130.5 |

**All null or negative**, including the two conditioning-architecture rewrites
(`orc_ctx_*`), which touched *how* the chemistry reaches the network rather than
what it contains. Nothing about presentation helps. My prediction that
hypernetwork/FiLM routing would help was falsified.

---

## 4. The one large positive result: the auxiliary loss

If the network will not compute the frame map from the TD signal alone, make it
an explicit target. `aux_canon_target` appends a 21-dim **supervision label**
(9 latent stone coordinates + 12 latent potion types) that is excised before the
observation ever reaches `RNN_head`, the critic, or the action mask. An
auxiliary head is trained against it.

Touched: **the training signal only.** No new input.

| run | aux weight | return | potion acc |
|---|---|---|---|
| `orc_sp_base` | off | 156.5 | — |
| `orc_aux_w0p1` | 0.1 | 229.8 | 0.703 |
| `orc_aux_w1` | 1.0 | **233.1** | 0.535 |
| `orc_aux_w10` | 10.0 | 233.9 | 0.549 |

Verified from `wandb/offline-run-*/logs/debug.log`: these run with
`canonicalize_oracle=False, context_graph_only=False`. **A training label alone
buys +77, reproducing what the two privileged observation rewrites bought
(+76).** This is the project's strongest result and it confirms the section-2
diagnosis exactly: the information was always there; only the incentive to
extract it was missing.

---

## 5. The same loss on MATE, and the asymmetry

For a memory model the same 21-dim target *is* privileged (MATE has no
`chem_gt`), so this is a different question — an upper bound, not a fair
method. It still asks something real: **if MATE is handed a dense signal to
learn the chemistry, does its return rise?**

| run | site | aux weight | return | potion acc | adaptation |
|---|---|---|---|---|---|
| `mem_mate_sp_s42` | — | off | **150.4** | — | 0.90 |
| `matep_aux_w0p1` | joint | 0.1 | 149.2 | 0.256 | 1.22 |
| `matep_aux_w1` | joint | 1.0 | 122.6 | 0.567 | -0.23 |
| `matep_aux_w10` | joint | 10.0 | 67.0 | 0.573 | -0.80 |

The opposite of the oracle. On the oracle the loss is worth +77; on MATE it
costs -28 at weight 1 and -83 at weight 10. **This asymmetry is the core
puzzle.** Two candidate causes were tested.

### 5.1 Was it the shared trunk? (Partly.)

`aux_canon_site` moves the head off the critic's own input.

| site | what the head reads | gradient reaches |
|---|---|---|
| `joint` | `conditioner(obs, h)` | encoder + conditioner + memory |
| `memory` | `h_t` only | memory only |
| `memory_obs` | `cat(obs.detach(), h_t)` | **memory only**, but can read the frame |

Touched: **where the auxiliary gradient lands.** Verified by gradient-flow test
(`/tmp/check_aux_site.py`): under `memory_obs` the conditioner and the observation
norm receive nothing.

| run | site | weight | return | potion acc |
|---|---|---|---|---|
| `matep_aux_w1` | joint | 1 | 122.6 | 0.567 |
| `mem_site_w1` | memory | 1 | 140.9 | **0.261** |
| `mo_site_w1` | **memory_obs** | 1 | **146.3** | **0.553** |
| `matep_aux_w10` | joint | 10 | 67.0 | 0.573 |
| `mo_site_w10` | memory_obs | 10 | 126.2 (partial) | 0.495 |

`memory` alone is inert — it cannot see the current frame, so it cannot learn
the map (0.261). `memory_obs` reads the frame but sends no gradient into it, and
recovers **+23.7 of the 27.8 points the shared trunk cost, at the same accuracy**
(0.553 vs 0.567). The residual -4.1 is within single-seed noise. At weight 10 the
collapse is eliminated (+59).

Accuracy trajectory for `mo_site_w1` across 8 buckets:
0.336 -> 0.383 -> 0.387 -> 0.429 -> 0.507 -> 0.532 -> 0.543 -> **0.551**. Still
climbing at the end.

**Conclusion.** The interference was real and is now fixed. But the *goal* was
not reached: chemistry knowledge that costs nothing still buys nothing. 146.3 vs
150.4 aux-off.

### 5.2 Was it exploration or action budget? (No.)

| # | intervention | touched | result |
|---|---|---|---|
| 5.2a | gradient clip 0.2 -> 12.0 (60x) | optimization | +0.4 pts. Rejected. |
| 5.2b | NO_OP action masking | action space | all 3 runs worse; control -24.5. Rejected. |

The NO_OP result is worth keeping: cash rate pinned at exactly 0.150 = 3 stones
/ 20 steps in all three masked runs, i.e. the mask forced cashing without
improving *which* stones got cashed.

Budget arithmetic from training-rollout rates x 200 steps (approximate; includes
epsilon exploration): MATE uses ~103 of 120 potion slots (86%) but cashes only
~16 of 30 stones (53%); the oracle cashes ~22 (73%). **Experimenting is not
scarce. Converting stones into +15 is.**

---

## 6. The open question and the experiment now running

MATE knows the potion map at 0.553 and gains nothing from it. Two explanations,
which MATE's own numbers cannot distinguish:

- **(a) 0.553 is too inaccurate to plan with.** Planning chains facts, so
  accuracy multiplies: a 3-potion route is right about 0.55^3 ~ 0.17 of the time.
- **(b) 0.553 would suffice** and the policy simply fails to act on it.

`canon_potion_acc` (this commit) separates them by running the *oracle* at
MATE's accuracy. It corrupts the canonicalization's potion half so each latent
type is reported correctly with probability p, with the wrong map drawn once per
episode and held fixed — a consistent wrong belief, not averageable noise. At
p=1.0 the path is bitwise identical to the uncorrupted one, so the 232.3
baseline still applies.

| p | prediction under (a) | prediction under (b) |
|---|---|---|
| 0.553 | collapses toward 145.2 | stays above 200 |
| 0.75 | intermediate | stays above 200 |

Only the potion half is degraded; stone coordinates stay exact. So this is an
**upper bound** for a memory model at the same accuracy. A collapse is
conclusive; survival is not.

**If (a):** the next job is raising 0.553, and the memory axis is the real
bottleneck after all. **If (b):** the next job is the policy/representation
side, and more accurate memory would be wasted.

### 6.1 The answer: (a). Accuracy is the binding constraint.

| p | return | gain over the 145.2 floor | share of the gain kept |
|---|---|---|---|
| 1.0 | 232.3 | +87.1 | 100% |
| **0.553** | **~170** | **+24.8** | **28%** |

Degrading only the potion half to MATE's accuracy destroys **72% of the gain**,
and that 170 is generous: this oracle still receives the bottleneck graph and
the stone coordinates exactly, neither of which MATE has. So a memory model at
0.553 is bounded by something *below* 170, and MATE already scores 150.4.

**This closes the section-5 puzzle, and it does so by discarding the ceiling
the puzzle was built on.** "MATE knows the chemistry at 0.553 and gains nothing
from it" was measured against 232.3, a ceiling reachable only at ~100% accuracy.
Against the correct ceiling the residual is under 20 points, not 82. MATE is
not failing to *use* what it knows; it does not know enough. The auxiliary loss
did not "fail to transfer" -- at 0.553 there was almost nothing there to
transfer.

The job is therefore **raising 0.553**, which is the memory axis.

---

## 6.2 The instrument the next round needs

Two gaps made the round above harder to read than it should have been, and both
are now closed in code.

**Per-trial accuracy** (`aux_canon_potion_acc_trial0..9`). 0.553 is an average
over all 200 steps of an episode. A flat 0.553 at every trial (the memory never
fills) and a 0.30 -> 0.80 climb (it does) report the same average and imply
opposite next experiments. Nothing in the logs could tell them apart.

**A measurement that does not perturb.** Every accuracy number so far came from
a head whose gradient was also *training* the thing it measured, so "what does
MATE know" could not be asked of a run that was not being taught the answer.
`aux_canon_site="probe"` is `memory_obs` with the memory detached as well: the
head trains, the agent does not. Verified by `scripts/verify_aux_sites.py`.

## 6.3 What is running now: the paper's own auxiliary task

The Alchemy paper's §4.3 tried three auxiliary tasks. The one this project
implemented -- predict the ground-truth chemistry -- is their task (3), the one
they report helped **least**. Their tasks (1) and (2), "Predict: Features", are
counts over the *current observation*: how many stones of each perceptual
category, how many potions of each colour. Those two took symbolic Alchemy
"close to the ideal observer benchmark", and the paper calls it

> "the only case in the present study where agents showed respectable
> meta-learning performance ... without privileged information at test."

`config_rl.aux_count_weight` implements it. The targets are a deterministic
function of the agent's own observation
(`envs.alchemy.count_targets_from_observation`, verified against the env's
internal chemistry by `scripts/verify_count_targets.py`), so nothing is added
to the observation and nothing can leak. The head attaches to the conditioner's
observation branch -- *not* `encoded_obs`, which for a non-pixel env is the
identity and shares no parameters with anything.

Why it is the right shape for this task: every measured gain here came from
making the observation easier to READ (`canonicalize_oracle` +35,
`structured_potions` +32) while capacity did nothing (+3.4 for 4x width, and a
wider net was negative). A per-category count cannot be produced without
reading every slot the same way and summing -- the permutation-invariant
structure a flat MLP over concatenated slots never acquires. It is
`structured_potions` asked for through the loss instead of installed in the
input layout.

| run | memory | count weight | probe | asks |
|---|---|---|---|---|
| `mate_probe_ctrl` | MATE | off | yes | what does MATE know from RL alone, per trial? |
| `mate_count_w1` | MATE | 1.0 | yes | does counting raise it? |
| `mate_count_w0p1` | MATE | 0.1 | yes | weight sensitivity |
| `oracle_count_w1` | oracle | 1.0 | — | does the lever reproduce on our oracle (vs `orc_sp_base` 156.5)? |

## 6.4 Result: the counting loss does not transfer. It is neutral or harmful.

All four at 94/94 evals, 4.81e6 env steps, seed 42.

| run | count weight | return | normalized | vs its control |
|---|---|---|---|---|
| `mate_probe_ctrl` | off | **153.0** | +0.055 | — (ledger `mem_mate_sp_s42` 150.4) |
| `mate_count_w0p1` | 0.1 | **153.0** | +0.055 | **0.0** |
| `mate_count_w1` | 1.0 | **107.8** | −0.263 | **−45.2** |
| `oracle_count_w1` | 1.0 | **135.7** | −0.067 | **−20.8** (vs `orc_sp_base` 156.5) |

**Not an implementation failure — the task is solved to near-perfection.**
Potion-count error falls 1.37 -> 0.006 and stone-count error 1.72 -> 0.059,
i.e. under a hundredth of a slot miscounted out of twelve. The network learns
to count and gains nothing for it.

**Why weight 1 hurts.** Early in training the auxiliary term is twice the
critic term (`aux_count_loss` 0.267 vs `critic_loss` 0.133), so the shared
observation branch is shaped around counting before the value function has any
say. By the time counting is solved and its loss has decayed to the critic's
own scale, the run is already behind and never recovers:

```
mate_probe_ctrl   return by 6ths:  99.4  141.7  144.7  149.2  149.2  153.0
mate_count_w1     return by 6ths:  67.7   99.7  102.8  106.2  107.6  107.8
oracle_count_w1   return by 6ths: 116.6  143.9  140.8  137.0  134.9  135.7
```

The oracle row is the clearest: it peaks at 143.9 and then declines
monotonically. That is an auxiliary objective progressively taking over a
shared representation, not a slow start.

**Why the paper's result did not carry over (best available reading).** In 3D
Alchemy "count the potions of each colour" requires parsing pixels into
objects, which is most of the perception problem. In symbolic Alchemy the same
target is a near-linear read of the slot block -- the measured error goes to
0.006. An auxiliary task teaches a representation only in proportion to how
much work it demands, and here it demands almost none while still competing for
the trunk.

## 6.5 What the probe found, which is the real result

`mate_probe_ctrl` is the first measurement of MATE's chemistry knowledge taken
with no gradient flowing into the agent. Three findings, none of them previously
measurable.

**(1) Unaided, MATE barely clears the memoryless ceiling.**

| | potion accuracy |
|---|---|
| chance | 0.1667 |
| memoryless ceiling (`scripts/probe_frame_map.py`) | 0.1675 |
| **MATE from RL alone (`mate_probe_ctrl`, measured)** | **0.222** |
| MATE when the aux loss explicitly teaches it (`mo_site_w1`) | 0.553 |
| needed for the return ceiling to exceed ~170 (§6.1) | ~1.0 |

The 0.553 this log has been quoting was obtained while *training on the
labels*. Without that teaching the figure is 0.222. The memory axis is in worse
shape than section 5 implied.

**(2) Knowledge does not accumulate within an episode. It decays.**

Per-trial accuracy across one 10-trial meta-episode (`mate_probe_ctrl`):

```
t0    t1    t2    t3    t4    t5    t6    t7    t8    t9
.244  .246  .242  .234  .215  .209  .204  .203  .205  .208
```

After 200 steps of using potions and observing outcomes, the memory is *less*
decodable than before any of it happened. A running mean over transition
embeddings is supposed to make exactly this quantity rise. It is the sharpest
evidence yet against the memory as currently wired, and nothing in the project
had measured it.

**(3) Accuracy falls as training proceeds**, identically in all three MATE runs:

```
accuracy by 6ths:  0.291  0.259  0.245  0.235  0.225  0.222
```

The memory is most informative when the policy is nearest to random.

**Two readings of (2)+(3), not yet separated.**
- *Exploration collapse* -- the memory stores fine, but the converged policy
  stops running informative experiments. §5.2 argued against this from potion
  usage (86% of slots consumed), but consuming a potion is not the same as
  learning from it: repeating one type on one stone spends the slot and returns
  no new information.
- *Dilution* -- the transition fed to MATE is 193 dims (`o_t, a_t, r_t,
  o_{t+1}-o_t` over a 76-dim observation) of which roughly ten carry the
  experiment's result; the other slots churn every step. Averaging that is
  averaging mostly noise.

The cheap discriminator: attach the probe to trajectories from a RANDOM policy
and read the per-trial curve. Rising means exploration collapse; flat means
dilution, and the fix is to build the transition from the acted-on operands
rather than the whole observation.

## 6.6 That discriminator ran, and its premise was wrong

`scripts/probe_memory_accumulation.py` holds the memory fixed (the
`mate_probe_ctrl` checkpoint) and swaps only the data: 512 episodes from the
trained policy vs 512 from a uniform-over-legal-actions policy. A fresh probe is
fit offline on each, held out by episode so it must generalise to chemistries it
never saw.

Neither curve rises, because **there is no signal for a curve to have a slope
in.**

| probe input | data | held-out | in-sample |
|---|---|---|---|
| obs only | trained policy | 0.183 | 0.714 |
| obs + memory | trained policy | **0.177** | 0.777 |
| obs only | random policy | 0.180 | 0.832 |
| obs + memory | random policy | **0.179** | 0.633 |

Against a memoryless ceiling of 0.1675 and chance of 0.1667. **Adding the
memory never helps, under either policy.** The result is unchanged across 128 vs
512 episodes, 3k vs 20k probe steps, standardized vs raw probe inputs, and
final vs early-stopped weights.

**The probe is not broken — a positive control settles it.** Asked to decode the
TRIAL INDEX from the memory alone, the same probe scores **1.000 held-out**
against a chance of 0.100.

> MATE's memory is a perfect clock and carries essentially no chemistry.

That is exactly what the readout statistics say independently: norm 2.51 per
step, but only **0.031 of across-episode standard deviation — 1.2%**. The
running mean is dominated by elapsed time and a near-constant mean transition
embedding; the chemistry-dependent component is roughly a hundredth of the
vector and does not survive as anything a decoder can generalise from.

**`aux_canon_potion_acc` is not trustworthy as reported.** The in-run metric is
scored on the batch the head just trained on, and over 480k updates that head
sees each buffered episode ~1000 times. Offline, a probe with **no memory at
all** reaches in-sample 0.71-0.85 while generalising at 0.18: it identifies the
episode from the observation and memorises that episode's map. So the 0.222 of
§6.5 and the 0.553 of §5 both contain a memorisation component of unknown size,
and neither is a measurement of transferable chemistry knowledge. Any future
use of this metric must report a held-out number.

**Limits of this result.**
- One checkpoint, one seed.
- It measures a MATE that was never asked to encode chemistry: `mate_probe_ctrl`
  has no memory-shaping loss, and the probe is detached by construction. Whether
  the architecture *can* encode chemistry when explicitly trained to is a
  separate question, answerable from a `mo_site_w1`-style checkpoint (not
  available in this repo's logs).
- The decoder is an MLP. That is the right decoder to test with, because the
  critic that must consume the memory is also an MLP.

**What it changes.** The question is no longer "why does accumulated chemistry
not get used" but "why is chemistry not in the memory at all". Exploration is
exonerated: better data through the same memory changes nothing. The dilution
account survives and now has a measurable target -- raise the chemistry share of
the memory's across-episode variance above 1.2% -- along with two metrics to
check it: that variance ratio, and held-out probe accuracy.

## 6.7 The memory CAN hold chemistry. It just has to be aimed at.

§6.5 measured a MATE that was never asked to encode chemistry. Three runs with
`aux_canon_site="memory_obs"` -- the head reads the frame but sends gradient
only into the memory (verified, `scripts/verify_aux_sites.py`) -- ask whether it
can be taught. Verdict from an offline HELD-OUT probe on each checkpoint, never
from the runs' own in-sample metric.

| run | aux parts | weight | probe: obs only | **probe: obs + memory** |
|---|---|---|---|---|
| `mem_w1_both` | both | 1 | 0.182 / 0.175 | **0.199 / 0.206** |
| `mem_w1_potion` | potion | 1 | 0.188 / 0.181 | **0.432 / 0.461** |
| `mem_w10_potion` | potion | 10 | 0.188 / 0.180 | **0.438 / 0.458** |

(two figures per cell = trained-policy data / random-policy data; chance 0.1667,
memoryless ceiling 0.1675. `mem_w10_potion` was probed twice independently and
returned 0.433 / 0.458 the second time.)

**Yes -- 0.43-0.46 against a 0.1675 ceiling, far past the 0.25 decision line
fixed before the runs.** This is the project's first trustworthy (held-out)
evidence of chemistry in a memory readout; every previous figure (0.553, 0.222)
was in-sample and contaminated.

**What mattered was WHERE the gradient went, not how hard it was pushed.**
Weight 1 -> 10 moves nothing (0.432 -> 0.438). Changing which half is scored
doubles the result (0.20 -> 0.43). The stone half is nearly free from one frame
(0.756 memoryless), so `parts="both"` spends most of the gradient on a problem
the memory is not needed for. Earlier memory-site experiments in this log used
the default `both` and were handicapped by it.

**And it accumulates within the episode** -- the shape §6.6 looked for and did
not find:

```
                        t0     t1     t2     t3     t4     t5     t6     t7     t8     t9
taught  (held-out)     .281   .389   .415   .433   .447   .455   .444   .479   .463   .435
untaught (§6.5)        .244   .246   .242   .234   .215   .209   .204   .203   .205   .208
```

Memory contribution over the obs-only control climbs +0.105 -> +0.261
(+0.0123/trial; +0.0168 on random-policy data). The first demonstration in this
project that MATE's running mean performs within-episode meta-learning at all.

**Capacity is genuinely reallocated.** At weight 10 the positive control -- the
trial-index clock that read 1.000 in §6.6 -- falls to 0.70. The memory gave up
clock precision to make room for chemistry.

**And the return still does not rise.**

| run | held-out accuracy | return |
|---|---|---|
| control (`mate_probe_ctrl`, aux off) | 0.177 | **153.0** |
| `mem_w1_both` | 0.199 | 151.3 |
| `mem_w1_potion` | 0.432 | 146.4 |
| `mem_w10_potion` | 0.438 | 137.5 |

Chemistry knowledge rises 2.6x and the score falls. That is not a contradiction:
§6.1 measured a return ceiling of ~170 at accuracy 0.553, so the ceiling at 0.43
is lower still, and the auxiliary loss charges interference on top.

**Open, and running:** an oracle at `canon_potion_acc=0.43` gives the return a
0.43-accurate agent can reach with a perfect graph and no aux interference. At
~155 the MATE runs are already at their ceiling and the only path is higher
accuracy; at 175+ the accuracy suffices and the failure is on the policy side.
Existing points: 1.0 -> 232.3, 0.553 -> ~170.

Caveat on that comparison: the knob holds accuracy CONSTANT, while MATE climbs
0.28 -> 0.48 across the episode. A flat 0.43 is generous early and harsh late.

## 6.8 The calibration came back at 167.7, and it closes the accuracy question

`canon_potion_acc=0.43` -- an oracle told the potion map at exactly the accuracy
MATE reaches, with a perfect graph and exact stone coordinates otherwise:

| potion-map accuracy | return |
|---|---|
| 1.00 | 232.3 |
| 0.553 | ~170 |
| **0.43** | **167.7** |

0.43 and 0.553 are the same number. **In this range, knowing more of the potion
map does not buy return.** The curve is steep only near 1.0.

## 6.9 Four interventions, four nulls

Each was aimed at a different link in "the memory knows it but the behaviour
does not show it". All against `mem_w1_potion` = 146.4 (memory at 0.43).

| run | intervention | return |
|---|---|---|
| `mate_factored` | dueling + factored Q head over NO_OP + stones x targets | **147.1** |
| `mate_feed` | the decoded potion posterior concatenated onto the critic input | **147.3** |

`mate_feed` is the decisive one: the critic was **handed the decode already
done**, detached, at both training and action-selection time. +0.9. Whatever
stops the policy, it is not that the critic cannot read chemistry out of the
memory.

And `scripts/diagnose_exploration.py` rules out the behavioural stories.
Trained MATE vs uniform-over-legal on the same checkpoint:

| | untaught (0.18) | taught (0.43) | uniform |
|---|---|---|---|
| repeat rate | 21.8% | 20.6% | 18.3% |
| potion wasted, axis ALREADY at that value | 49.9% | **49.2%** | 49.8% |
| potion wasted, edge BLOCKED | 11.4% | 11.3% | 10.0% |
| stones cashed | 13.5/30 | 13.0/30 | 29.6/30 |
| jackpot stones abandoned | **0%** | 0.3% | 15.4% |

- Exploration is not degenerate (repeat rate at uniform's level).
- The cash policy is already correct -- it never abandons a +15 stone and
  discards only negatives, which is optimal.
- **Teaching the memory to 0.43 changes the behaviour by 0.7 percentage
  points.** The useless-potion rate is uniform's.

A correction on the way: the first version of that script attributed all 60.9%
of no-change results to blocked edges. Splitting them shows only 11% are the
graph; the other 50% are "the axis is already at that value", which is a
potion-map-and-position fact, not a graph fact. The blocked-edge counter was
also wrong (18.8 of 12) -- it counted non-adjacent vertex pairs. Real figure:
~2.8 of 12 edges closed per episode.

## 6.10 It was never the potion map. It is the FRAME.

The 2x2 that should have been run first. Oracle, `aux_canon_site="joint"`,
weight 1, everything else identical -- only which half of the label is scored:

| | `parts=both` (stones + potions) | `parts=potion` |
|---|---|---|
| **DQN oracle** | **233.1** | **97.8** |
| **SAC-discrete oracle** | **160.4** | 161.1 |

Dropping the stone half costs DQN **135 points** and lands it 47 BELOW the
no-chemistry floor. So §4's celebrated +77 was never the potion supervision:
it was the nine stone-coordinate regressions, which are the instruction
"compute the perceived -> latent frame map".

Three independent measurements now say the same thing:

| lever | effect |
|---|---|
| `canonicalize_oracle` (env does the frame alignment) | +34.4 |
| potion-map accuracy 0.43 -> 0.553 (§6.8) | 167.7 -> 170 |
| aux `both` -> `potion` (removes the frame signal) | **233.1 -> 97.8** |

**The frame is where the value is. The potion map on its own is worth almost
nothing.** Every experiment in §6.5-6.9 optimised potion-map accuracy and is
therefore measuring the wrong quantity; that includes the held-out 0.43 result,
which stands as a fact about the memory but not as progress on return.

## 6.11 The algorithm axis exists, but only under the auxiliary loss

| | no aux | aux, `parts=both` |
|---|---|---|
| DQN oracle | 156.5 | **233.1** |
| SAC-discrete oracle | 159.9 | **160.4** |

Without the aux loss the two learning rules are 3.4 points apart -- §6.11's
first row is what justified "156.5 is an algorithm-independent ceiling". With
it they are **73 points** apart. Learning the frame map from a dense label is
something DDQN does and this SAC-discrete does not.

On the MEMORY axis the algorithm is irrelevant:

| | DQN | SAC-discrete |
|---|---|---|
| MATE, no aux | 153.0 | **153.1** |
| MATE + aux (`potion`) | 146.4 | **154.3** |

Identical without the aux loss. The difference is what the aux loss COSTS:
DQN pays -6.6 for it, SAC-discrete does not (+1.2).

**No further MATE + `parts=both` run is needed** -- both sites have been
measured and neither beats the 153.0 control: `joint` gives 122.6
(`matep_aux_w1`, §5) and `memory_obs` gives 151.3 (`mem_w1_both`, §6.7). For
the oracle the stone label says "compute a map from inputs you already hold";
for a memory model the same label is privileged and says "invent one". Same
bytes, opposite effect.

## 6.12 SAC-discrete: revived, and what it cost

The handover note (`alchemy_next_experiments.md` P2a) called this half a day of
shape fixes. It was five traps, three of which never raise:

| # | trap | crashes? |
|---|---|---|
| 1 | `sac_default.py` lacks the CLI keys | yes |
| 2 | `target_entropy` defaults to the CONTINUOUS `-action_dim`; discrete needs `+0.98 log A` | **no** |
| 3 | with action masking, `0.98 log(40)=3.62` exceeds `log(legal)=1.39` -- unreachable, alpha diverges | **no** |
| 4 | `sample_random_action` missing (only called once masking is on) | yes |
| 5 | ratio 0.98 pins entropy at 98% of maximum, so a deterministic argmax eval reads noise -- return 0.00 for 19 straight evals | **no** |

Trap 5 only appeared in training: alpha sat at a healthy 0.91 and the dual
tracked its target exactly. The TARGET was the problem. At ratio 0.3 return
went 0.00 -> 12.3 -> 28.8 -> 49.1 and the run converged normally.

One approximation is mine and not in Christodoulou 2019: alpha is a single
scalar while the legal-action count varies 1..40, so the target is the batch
mean of `0.98 log(n_valid)`. States with one legal action drag it down.

---

## 7. Corrections to earlier claims in this log's history

Recorded because each one changed a conclusion.

- **"MATE 0.518 accuracy ~ oracle 0.535 accuracy, yet 92 points apart, so the
  bottleneck is use not memory."** Wrong. The oracle *receives* the chemistry as
  an input, so it knows it at 100%; its 0.535 is a separate decoder's readout and
  does not bound its planning. MATE's 0.553 is its actual knowledge ceiling. The
  correct comparison is 100% vs 55%, which is why section 6 exists.
- **Mid-run adaptation over-read.** `mo_site_w1` showed adaptation 1.22 mid-run;
  the final value is 0.73, *below* the aux-off control's 0.90, while the inert
  `mem_site_w1` scores 1.51. Adaptation is too noisy at this sample size to
  support conclusions.
- **"MATE's memory cannot hold chemistry" (§6.6 reading).** Overturned by §6.7.
  It holds it at 0.43 held-out when the auxiliary gradient is aimed at the
  potion half. §6.6's finding stands as written -- an UNTAUGHT memory holds
  none -- but the dilution account it favoured is no longer the leading
  explanation, and "the running mean is structurally unable" is refuted.
- **Prediction track record: 1 of 8.** Falsified: hypernetwork/FiLM routing,
  learning-rate tuning, aux-loss transfer to MATE, the clipping confound, NO_OP
  masking, the "Predict: Features" counting loss (§6.4 -- predicted to
  reproduce the paper's largest gain; measured 0.0 at weight 0.1 and −45.2 at
  weight 1.0), and the prediction that a taught memory would also come back at
  ~0.18 (§6.7 -- it came back at 0.43). Held: the oracle aux-loss prediction.
- **"MATE knows the chemistry at 0.553."** Superseded. That number was measured
  while the aux loss was training on the labels. Measured with an inert probe
  (§6.5) the unaided figure is **0.222**, against a 0.1675 memoryless ceiling.

---

## 8. Known gaps

- `mem_lstm_sp_s42` (17/94) and `mem_gpt_sp_s42` (14/94) died of OOM. Buffers
  are preserved; the architecture comparison is unfinished and its numbers above
  are short-run and not comparable to the 94-eval rows.
- All headline numbers are single-seed. The two oracle seed pairs that exist
  (`oracle_canon_s42`/`s43` = 193.5/192.0, `oracle_full_p0`/`s43` = 232.3/231.4)
  suggest ~1-2 points of seed noise on the oracle, but MATE-side variance is
  unmeasured.
- Deferred: widening MATE memory 256 -> 512; logging the cashed-stone value
  distribution (which would show directly whether the missing return is
  uncashed +15s or cashed -1s).
