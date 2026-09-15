# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Memory-RL is an experimental testbed for **MATE** (Memory of Accumulated Transition
Embeddings), a memory-based RL framework for Contextual MDPs (POMDPs). A sequence of
transitions `(s, a, r, s')` is processed by a sequence model and the result is used as
memory for decision-making. Unlike Transformers or RNNs, MATE's memory is a *running
normalized sum* of transition embeddings — bounded memory, still context-sensitive.

**Environments:** T-Maze (passive/active), MuJoCo (cheetah-vel, ant-dir, hopper-param,
walker-param), Metaworld (ML10/ML45), Symbolic Alchemy, CARL Vehicle Racing (pixels)
**Algorithms:** DQN (discrete), SAC (continuous) — both with optional PopArt
**Memory architectures:** MATE (+MSC variants), SplAgger, GPT-2, LSTM, GRU, RNN, Markov / oracle Markov

## Setup

The working env on this machine is **`hist`** (python 3.10) — use it for every run:

```bash
conda activate hist
```

To recreate it elsewhere:

```bash
conda create -y -n hist python=3.10
conda activate hist
pip install -r requirements.txt
```

Symbolic Alchemy needs `dm_alchemy`, which is **not on PyPI** (archived, GitHub-only) and
is deliberately kept out of `requirements.txt`:

```bash
bash scripts/install_dm_alchemy.sh hist
```

CARL Vehicle Racing needs `Box2D` + `pygame`, also kept out of `requirements.txt`
(`conda install -c conda-forge swig && pip install "gymnasium[box2d]" pygame`; see `readme.md`).

MuJoCo rendering backend: `export MUJOCO_GL=glfw` (windowed) / `egl` (headless GPU) /
`osmesa` (headless CPU).

## Running Training

All three config flags are required.

```bash
python main.py \
  --config_env configs/envs/<env>.py --config_env.env_name <name> \
  --config_rl configs/rl/<dqn|sac>_default.py \
  --config_seq configs/seq_models/<mate|gpt|lstm|markov|splagger|mate_msc*>_default.py \
  --train_episodes <N> --device <gpu_id> --run_name <experiment_name>
```

```bash
# T-Maze passive, corridor length 100, MATE + DQN
python main.py --config_env configs/envs/tmaze_passive.py --config_env.env_name 100 \
  --config_rl configs/rl/dqn_default.py --config_seq configs/seq_models/mate_default.py \
  --train_episodes 20000 --device 0 --run_name tmaze_mate

# MuJoCo cheetah-vel, GPT-2 + SAC
python main.py --config_env configs/envs/mujoco.py --config_env.env_name cheetah-vel \
  --config_rl configs/rl/sac_default.py --config_seq configs/seq_models/gpt_default.py \
  --train_episodes 25000 --device 0 --run_name test

# Pixel env (CARL) — enable the CNN encoder
python main.py --config_env configs/envs/carl_vehicle_racing.py --config_env.env_name all \
  --config_rl configs/rl/sac_default.py --config_seq configs/seq_models/mate_default.py \
  --config_seq.use_image_encoder=True --train_episodes 10000 --device 0 --run_name test
```

**Top-level flags** (`main.py`): `--seed`, `--device`, `--batch_size`, `--train_episodes`,
`--start_training`, `--updates_per_step`, `--freeze_critic`, `--save_dir`, `--run_name`,
plus:
- `--max_seq_len N` — number of *real* transitions per training update. `-1` (default)
  uses the full episode. Inference always uses the full history. Sampling mode follows
  `config_seq.seq_model.use_store`: **STORE** takes a sorted random *subset* of transitions,
  everything else a contiguous BPTT *window*. There is no separate sampling-mode flag.
- `--save_buffer` (default `True`) — write `buffer_checkpoint.pth`. It is ~99% of checkpoint
  time, so `False` saves 3-6% of wall time; `--resume` then starts from an empty buffer.
- `--resume <log_dir>` — resume from `training_checkpoint.pth`; `utils/experiment.py::validate_resume_config`
  rejects any config drift except `schedule_steps` / `replay_buffer_num_episodes`. The buffer is
  restored only if `buffer_checkpoint.pth` exists; otherwise the `--start_training` warm-up re-runs.
- `--timestamp <str>` — pin the log-dir timestamp (e.g. `AMLT_EXPERIMENT_NAME`) so preempt-resume
  lands in the same directory.

Use `--config_section.flag=value` (with `=`) for `ml_collections` overrides, **especially
booleans** — `--config_env.visualize_env False` silently does the wrong thing.

## Testing

There is **no `tests/` directory in this repo**, despite `readme.md` and
`.github/copilot-instructions.md` documenting a `python -m unittest discover -s tests`
runner. Validation is manual: the root Jupyter notebooks (`check_tmaze_embedding.ipynb`,
`TSNE_visualization.ipynb`, `return_graph_vis.ipynb`, `check_time_vis.ipynb`) and short
training runs. If you add tests, use the stdlib `unittest` runner named above so the
existing docs become true.

`budget.py` is outdated — never use it as an implementation reference.

## Architecture

### 3-Part Config System

| Flag | Location | Controls |
|------|----------|----------|
| `--config_env` | `configs/envs/*.py` | Environment, episode length, `n_env`, eval/log intervals |
| `--config_rl` | `configs/rl/*.py` | DQN/SAC, learning rates, discount, PopArt, replay size |
| `--config_seq` | `configs/seq_models/*.py` | Memory architecture, conditioning, dropout, image encoder |

Each family starts from its `common.py` `base_config()`.

- Env configs define `create_fn(config) -> (config, registered_env_name)`: it registers the
  Gymnasium env (including `max_episode_steps`) and `del config.create_fn` before returning.
- RL/seq configs define `update_fn`, invoked by `utils/experiment.py::finalize_training_configs`
  **only after** the real `max_episode_steps` is known (this is what sets
  `config_seq.seq_model.max_seq_length` and the DQN epsilon schedule). They also delete themselves.
- `main.py` resolves the configs → builds two non-autoresetting `AsyncVectorEnv`s (train/eval)
  → `wandb.init` → `Learner.train()`.
- `log_interval`, `eval_interval`, and `eval_episodes` must each be divisible by `config_env.n_env`
  (enforced in `validate_run_settings`).

### Component Hierarchy

```
main.py
└── Learner (policies/learner.py)  — rollout, replay updates, eval, W&B, checkpoints
    ├── AsyncVectorEnv × 2 (autoreset DISABLED), episodes assembled via policies/rollout.py
    ├── RolloutBuffer (buffers/rollout_buffer.py) + ObservationStore (ram | memmap)
    └── Agent: AGENT_CLASSES[config_rl.algo] → Policy_DQN_RNN | Policy_SAC_RNN
        │   (policies/models/policy_rnn_dqn.py / policy_rnn_sac.py — each owns its FULL
        │    algorithm: heads, losses, optimizers, targets, PopArt, torch.compile)
        └── RNN_head (policies/models/recurrent_head.py)
            ├── image_encoder: optional CNN (pixel envs)
            ├── encoded_obs_norm / transition_input_norm: InputNorm (normalize_inputs)
            ├── transition_embedder: Linear+LeakyReLU+Dropout — Identity for mate/markov
            ├── seq_model: SEQ_MODELS[name]  (mate, splagger, gpt, lstm, gru, rnn, markov)
            ├── optional sinusoidal PE on the memory readout (use_pe)
            └── conditioner: Concat | FiLM | Hyper  (policies/models/conditioning.py)
```

There is **no `policies/rl/` package and no `RL_ALGORITHMS` registry** — the DQN and SAC
implementations live entirely inside their agent classes. `AGENT_CLASSES` in
`policies/models/__init__.py` is the only dispatch point. Keep DQN/SAC batch prep and
gradient clipping shared through `policies/models/off_policy_utils.py`
(`RecurrentBatch`, `prepare_recurrent_batch`, `clip_gradients`).

### Replay layout & sequence alignment

Everything is time-major `(L, B, dim)` with an explicit dummy/context row at index 0:

- `act`, `rew`, `term`, `mask`: `(T+1, B, dim)`; `obs` and `obs2` are **separate** stores
  (not a shifted `(T+2)` obs tensor).
- `mask[0] == 0` always — the sample start acts as the `t=-1` reset point. Sequence models
  and losses depend on this; do not remove or shift it.
- `transition_t` `(L, B)` carries the *absolute* environment row index, so positional
  encoding and shared-state normalization stay correct even inside a sampled window.
- `random_episodes(batch_size, mode)` returns the full episode when `max_seq_len <= 0`,
  otherwise a contiguous `window` or a sorted random `subset` of `max_seq_len` transitions.
- Observations live in `buffers/observation_store.py`: `ram` (default, on the **GPU**) or
  `memmap` (on disk; the dir prefers `/scratch` when present). Both honor
  `config_env.obs_dtype` — use `"uint8"` for pixels (4x smaller; batches are cast to float32
  on sample). See `configs/envs/carl_vehicle_racing.py`.
- `cached_embeddings` / `cached_prefixes` are only allocated when `seq_model.use_store=True`
  (STORE; `cached_embeddings` holds per-row `z`, `cached_prefixes` its `cumsum` at sample time).

### RNN_head (`policies/models/recurrent_head.py`)

The sole consumer of top-level `config_seq.*` architecture flags and the single boundary
between replay data and every sequence model.

- **`forward(actions, rewards, observs, next_observs, masks, transition_t, …)`** →
  `(current_joint, next_joint, d_forward)`. It returns **explicit current and successor
  embeddings** for the Bellman backup (not one shifted `(T+2)` tensor). Row 0 is the masked
  context transition. `d_forward` is logged to W&B; `_aux_loss` is the reserved
  differentiable auxiliary-loss channel and `_cache_z` the refreshed embedding channel.
- **`step(prev_internal_state, prev_action, prev_reward, prev_obs, obs, initial, timestep)`** →
  `(joint_embed, current_internal_state, transition_embedding)` — single-step rollout (L=1).

**Transition input convention** (`_build_raw_transition`):
- `full_transition=True`: `(o_t, a_t, r_t, o_{t+1} - o_t)` — delta form
- `full_transition=False, obs_shortcut=True`: `(o_t, a_t, r_t)`
- `full_transition=False, obs_shortcut=False`: `(a_t, r_t, o_{t+1})`

**Transition embedder dispatch**: `markov` and `mate` get `IdentityModule()` with
`seq_input_size = transition_size` (markov ignores it; MATE owns its own projection inside
`Mate.embedder`). Every other model gets `Linear(transition_size, hidden_dim) → LeakyReLU →
Dropout(dropout_emb)`.

**Dummy-step handling**: with `obs_shortcut=True` the `t=-1` transition is dropped before the
seq model and the zero-internal-state hidden is prepended — for MATE via
`internal_state_to_hidden` (preserving the learned `init_emb` prior), for others a zero vector.

**Joint embedding** (`policies/models/conditioning.py`) — all three share
`forward(x, c) -> joint` and `.out_dim`, and `RNN_head.embedding_size = conditioner.out_dim`:
- `"concat"` → `ConcatConditioner`: MLP on `encoded_obs`, then `cat(out, h_t)`;
  `out_dim = mlp_out_dim + cond_dim`. `cond_dim = 0` for markov, so it degenerates to a plain MLP.
- `"film"` → `FiLMConditioner`: `Linear(in→h)` then `n_layer × (Linear → act → FiLM(·, h_t))`.
  `(γ, β)` heads zero-init so the stack starts as identity (Perez+ 2017, arXiv:1709.07871).
- `"hypernet"` → `HyperConditioner`: `Linear(in→h)` then `n_layer × (HyperLinear(·, h_t) → act)`,
  Hyperfan-In init.
- `conditioning_n_layer` counts blocks added *after* the plain input projection;
  `conditioning_hidden_dim` sets the conditioner width, decoupled from `seq_model.hidden_size`.
- `film`/`hypernet` are asserted non-markov. With `obs_shortcut=False` no conditioner is built
  and `joint_embed = h_t`.

**Other `config_seq` knobs handled here**: `use_pe` (absolute sinusoidal PE added to the memory
readout, scaled by a learned zero-init `pe_scale`; requires `seq_model.max_seq_length` and an
even `cond_dim` — for markov the readout is zero so PE *is* the conditioning signal),
`project_output` (project obs and memory readouts onto the radius-`sqrt(D)` hypersphere),
`noise_ratio` (Gaussian noise in normalized feature units; requires `normalize_inputs=True`).

**`_encode_obs` & oracle Markov**: with a CNN and `seq_model.is_oracle`, only the image prefix
goes through the encoder — the `context_dim` tail appended by `envs/wrapper.py::oracleWrapper`
must be preserved and re-concatenated. `context_dim` is discovered at runtime by
`Learner.init_env` from `info["context"]` and written back into `config_seq.seq_model`.

**`torch.compile`**: `config_seq.compile=True` lazily compiles only the agent's CUDA
training-loss graph. Rollout, optimizer/scheduler steps, and target updates stay eager.
Disable it when debugging shape/dtype issues — compiled-graph errors are noisy.

### MATE (`policies/seq_models/mate_vanilla.py`)

Internal state is `(cumsum, count)` with shapes `(1, B, hidden_size)` and `(1, B, 1)`.

```python
z = self.embed_transitions(inputs)                       # (T, B, hidden_size)
cumsum = torch.cat([hidden, z], dim=0).cumsum(0)[1:]     # NOT hidden + z.cumsum(0):
                                                         # avoids an Inductor SplitScan +
                                                         # broadcast crash (pytorch#180221)
counts = initial_count + arange(1, T+1).view(T, 1, 1)
output = cumsum / counts.clamp(min=1e-6)                 # running mean
```

`Mate.embedder` owns the whole `transition_size → hidden_size` pipeline: an input projection
`Linear(in→h) → LeakyReLU → Dropout(dropout_emb)` followed by exactly `n_layer` additional
`Linear(h→h) → LeakyReLU → Dropout(dropout_ff)` blocks (`n_layer=0` → projection only).

**`normalize_z`** (MATE-only, default off): an `InputNorm` on `z` inside `embed_transitions`,
so aggregation, the `init_emb` prior, MSC and the rollout z cache all operate on normalized
embeddings. Stats update only in training mode — from `Mate.forward`, or from
`contrastive_loss` in `alternating_ema` mode.

**Initial-memory prior.** With `learn_init_emb=True` (config default), `init_emb` and
`log_init_weight` give `m_t = (w·init_emb + Σ z_i) / (w + t)`; `get_zero_internal_state`
starts from that prior sum and count. With `learn_init_emb=False` both start at zero.
`use_ema_init_emb=True` makes `init_emb` a bias-corrected EMA buffer of valid training
embeddings instead of a learned parameter (requires `learn_init_emb=True`).

**STORE** — Subset Training Over REused Embeddings. Turn it on with
`--config_seq.seq_model.use_store=True` on top of any MATE config (`forward_cached`); there is no
separate config file. Each update recomputes only the `--max_seq_len k`
sampled transition embeddings and reuses the rollout-cached ones for the rest of the episode,
correcting them by `z - cached_z`, so the memory is the exact full-episode running mean while
backprop touches only `k` transitions. `use_store` is also what selects subset replay sampling
(`Learner.init_train` sets `rl_sample_mode`); it requires `obs_shortcut=True` and `k >= 2` and is
incompatible with MSC (all asserted there). Refreshed embeddings flow back to the buffer through
`d_forward["_cache_z"]`.

- **`store_grad_correction`** (default `True`): the loss reaches `z` only through pairs `(t, i<t)`
  — `next_joint` is consumed under `no_grad` in both agents — which survive subset sampling with
  probability `k(k-1)/(T(T-1))`, while every path that skips `z` survives with `k/T`. Under the
  shared `1/num_valid` normalization the embedder is therefore scaled down by `(k-1)/(T-1)`.
  `forward_cached` undoes this with a straight-through factor `α = (T-1)/(k-1)` on
  `correction_before` (the sole `z` gradient path), leaving the forward value untouched, so the
  update is an unbiased estimate of the full-episode gradient. `α = 1` when `k == T`, which is why
  the untruncated `window` fallback stays correct. The cost is ~`T/k` variance on that path.
- Why subset and not window: a contiguous window can never put a pair with gap `>= k` in one
  batch, so credit assignment stops at `k`; a random subset gives every gap positive probability.
  MATE has no gap decay at all (`∂m_t/∂z_i = 1/(w+t)` for every `i < t`), so this matters a lot —
  learnable context length stops being capped by the per-update budget `k`.
- Why MATE only: an RNN hidden state comes from an order-dependent recursion, so an arbitrary
  subset cannot be recomputed in place. MATE's memory is a sum, so individual terms can be swapped
  independently (`delta = z - cached_z`).
- Not fixed by `α`: the *value* of the reused prefix is stale for rows not sampled recently, and
  cached `z` predate later `z_norm` stat drift.

**MSC (contrastive aux)** — `policies/seq_models/msc_aux.py` (`legacy`) and `msc_v2_aux.py` (`v2`):
- `mate_msc_default.py` = legacy anchor-based InfoNCE; `msc_view` picks the positive-pair family
  (`subset` | `split` | `temporal` | `prefix` | `transition` — see the docstring in that config),
  and `msc_learn_gains` applies learned per-feature gains to the memory on the policy path.
- `mate_msc_v2_default.py` = two equal-size disjoint random transition subsets per episode with
  symmetric bilinear CPC, each using MATE's actual mean memory
  `(init_emb + Σ_{i∈S} z_i) / (init_weight + |S|)`. The init prior/count is detached;
  `msc_detach_z` controls only whether MSC gradients reach the encoder.
- `mate_msc_ema_v2_default.py` sets `msc_update_mode="alternating_ema"`: a separate optimizer
  trains the online `embedder` with CPC (`msc_updates_per_rl` steps per RL step, `msc_lr`) while a
  frozen EMA copy supplies the RL/rollout memory. Requires `msc_detach_z=False`.
- MSC is incompatible with STORE (`use_store`) in every variant, so no MSC run uses subset replay
  sampling (asserted in `Learner.init_train`).

### Adding a New Sequence Model

Implement an `nn.Module` with:
- `name` — class attribute, the registry key
- `hidden_size` — instance attribute; `RNN_head` uses it to decide whether to log hidden-norm
  stats (0 for no-memory models). Optionally expose `output_size` when the readout width
  differs from `hidden_size`.
- `forward(inputs, h_0, **kwargs) -> (output, h_n[, info])` — `info` is a dict merged into
  `d_forward` and logged to W&B; put a differentiable aux loss under `info["_aux_loss"]`.
- `get_zero_internal_state(batch_size, **kwargs) -> h_0`
- `internal_state_to_hidden(internal_state) -> (1, B, hidden)` — only called when
  `obs_shortcut=True` and `name == "mate"`; other models get a zero dummy prepended instead.

Optional kwargs some models consume: `mask`, `compute_msc`, `return_embeddings`, `obs_emb`.
Register in `SEQ_MODELS` in `policies/seq_models/__init__.py`.

### Environments

`envs/make_env.py` builds the registered env (or `MLWrapper` for `ML*`), then applies
`oracleWrapper` when `is_oracle`. Per-attempt adaptation curves come from an env's native
trials (`config_env.num_trials`, Alchemy only).

tmaze/mujoco/metaworld also accept `reset(options={"keep_context": True})` to hold the task
fixed across a reset. Nothing in-tree passes it (it fed the removed k-shot wrapper); it is kept
for meta-episode ablations.

## Logging & Checkpoints

W&B: entity `mate_research`, **project = the registered env string** (e.g. `tmaze_passive_T-100`),
run name = `run_name`. Local dir: `{save_dir}/{env_type}/{env_name}/{run_name}_{timestamp}/`.

```
training_checkpoint.pth   # versioned (CHECKPOINT_FORMAT_VERSION=2), agent + counters + W&B ids + configs
buffer_checkpoint.pth     # the whole replay buffer; skipped when --save_buffer=False
```

Agents must put model, optimizer, scheduler, target, and algorithm-specific state into
`training_state_dict()` and restore it in `load_training_state_dict()`. Pre-refactor
checkpoints are not loadable (`utils/checkpointing.py` raises on a version mismatch).
Per-timestep tensors are logged as matplotlib figures under `visualizations/` every
`visualize_every * log_interval` episodes (train) / `visualize_every * eval_interval` (eval).

### Adding metrics to `info` — avoid CPU-GPU sync

Anything you write into a `seq_model.forward` `info` dict or `RNN_head`'s `d_forward` must
**stay on the GPU**. The Learner batches the CPU transfer at log time; syncing per step
destroys throughput (commit `d710213` "eliminate GPU-CPU sync points" was the original fix).

**Do** (no sync):
```python
info["init_emb_norm"] = self.init_emb.detach().norm()             # 0-dim GPU tensor
info["memory_norm"]   = output.detach().norm(dim=-1).mean(dim=1)  # (T,) GPU tensor
```

**Don't** (forces a sync every forward):
```python
info["x"] = t.norm().item()      # .item() blocks
info["x"] = t.mean(dim=1).cpu()  # .cpu() blocks
print(f"norm = {tensor}")        # implicit .item()
if tensor > 0: ...               # implicit .item() on a 0-dim CUDA tensor
```

Reductions (`.mean`, `.norm`, `.std`, `.abs().max()`) stay on-device — only the final wandb
commit should sync, once per `log_interval`.

## AMLT jobs

Cluster job configs live in `amlt/*.yaml`. **Read `amlt/CLAUDE.md` before editing them** — it
documents the verified image/setup, the cluster/storage table, and the pitfalls below.

```bash
amlt run amlt/<config>.yaml <experiment_name>
amlt status <experiment_name>
amlt logs <experiment_name> :<job_name>
```

- AMLT expands YAML commands with Python `string.Template`: escape shell `$` as `$$`, leave
  AMLT substitutions like `${BASE_DIR}` single.
- Singularity jobs have **no sudo** — install system libraries with `conda install`.
- Headless MuJoCo works via OSMesa only: `mesalib<25.1` (newer Mesa ships no `libOSMesa.so`;
  `mesa` is an unrelated PyPI package), `MUJOCO_GL=osmesa`, `PYOPENGL_PLATFORM=osmesa`, and
  `LD_LIBRARY_PATH` pointing at the conda env's `lib`. GPU EGL is not reliable on this cluster.
