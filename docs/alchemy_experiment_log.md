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
- **Prediction track record: 1 of 6.** Falsified: hypernetwork/FiLM routing,
  learning-rate tuning, aux-loss transfer to MATE, the clipping confound, NO_OP
  masking. Held: the oracle aux-loss prediction.

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
