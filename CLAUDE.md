# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Memory-RL is an experimental testbed for **MATE** (Memory of Accumulated Transition Embeddings), a memory-based RL framework for solving Contextual MDPs (POMDPs). Unlike Transformers or RNNs, MATE maintains memory by summing transition embeddings — preventing unbounded memory growth while enabling context-sensitive behavior.

**Supported environments:** T-Maze (passive/active/detour), MuJoCo (cheetah-vel, ant-dir, hopper-param, walker-param), Metaworld (ML10, ML45)  
**Supported algorithms:** DQN (discrete), SAC (continuous)  
**Memory architectures:** MATE (vanilla + RFF variant), SplAgger, GPT-2, LSTM, GRU, RNN, Markov (no memory)

## Setup

```bash
conda create -y -n mate python=3.10
conda activate mate
pip install -r requirements.txt
```

For MuJoCo rendering:
```bash
export MUJOCO_GL=egl    # headless GPU
export MUJOCO_GL=osmesa # headless CPU
```

## Running Training

```bash
python main.py \
  --config_env configs/envs/<env>.py \
  --config_env.env_name <env_name> \
  --config_rl configs/rl/<dqn|sac>_default.py \
  --config_seq configs/seq_models/<mate|gpt|lstm|markov>_default.py \
  --train_episodes <N> \
  --device <gpu_id> \
  --run_name <experiment_name> \
  [--seed <seed>] [--batch_size <B>] [--updates_per_step <K>]
```

**Example — T-Maze with MATE:**
```bash
python main.py --config_env configs/envs/tmaze_passive.py --config_env.env_name 100 \
  --config_rl configs/rl/dqn_default.py --config_seq configs/seq_models/mate_default.py \
  --train_episodes 20000 --device 0 --run_name tmaze_mate
```

budget.py is outdated. Do not use it for reference.

There are no automated tests; Jupyter notebooks in the root (`check_tmaze_embedding.ipynb`, `TSNE_visualization.ipynb`, etc.) are used for analysis and validation.

## Architecture

### 3-Part Config System

Training requires three config files passed as flags:

| Flag | Location | Controls |
|------|----------|----------|
| `--config_env` | `configs/envs/*.py` | Environment type, episode length, eval intervals |
| `--config_rl` | `configs/rl/*.py` | Algorithm (DQN/SAC), learning rates, epsilon schedule |
| `--config_seq` | `configs/seq_models/*.py` | Memory architecture, hidden size, layers |

**`create_fn` vs `update_fn`:** Env configs define `create_fn(config) → (config, env_name)` which registers the Gymnasium environment. Seq/RL configs define `update_fn(config, max_episode_steps)` which computes derived parameters (e.g., `max_seq_length`). Both functions delete themselves from the config before returning. Config is loaded by `main.py`, passed to `Learner`, and shared across all components.

**Key seq config flags** (in `config_seq`):
- `obs_shortcut`: if True, observation is embedded separately and concatenated with the seq model output → joint embedding `(obs_embed, h_t)`
- `full_transition`: if True, transition input is `(o_t, a_t, r_t, o_{t+1})`; if False, uses only `(o_t, a_t, r_t)` or `(a_t, r_t, o_{t+1})`
- `project_output`: if True, L2-normalizes hidden states scaled by `sqrt(hidden_dim)`

### Component Hierarchy

```
Learner (policies/learner.py)
├── AsyncVectorEnv (n_env parallel envs)
├── RolloutBuffer (buffers/rollout_buffer.py)  — trajectory storage, optional obs/reward normalization
└── Agent (Policy_DQN_RNN or Policy_SAC_RNN)
    ├── RNN_head (policies/models/recurrent_head.py)  — core architecture
    │   ├── image_encoder: optional CNN, applied to obs (used for pixel envs)
    │   ├── transition_embedder: MLP or RFFEmbedding → hidden_dim
    │   ├── seq_model: MATE/MateRff/SplAgger/GPT2/LSTM/Markov processes embedded transitions
    │   └── observ_embedder: MLP(obs) → obs_embed (when obs_shortcut=True)
    ├── Critic: MLP over joint_embed = cat(obs_embed, h_t)
    └── RL Algorithm (policies/rl/dqn.py or sac.py)  — loss computation, action selection
```

### Data Flow

**At inference (one step):**
1. `Learner.act()` calls `agent.act(prev_obs, action, reward, obs, internal_state)`
2. `RNN_head.step()` embeds the transition, updates the seq model's internal state
3. Critic/actor selects action from joint embedding `(obs_embed, h_t)`

**At training (full trajectory):**
1. Sample batch of full episodes from `RolloutBuffer`
2. `agent.forward()` processes all timesteps through `RNN_head.forward()`
3. RL loss computed over valid steps (using masks); gradients flow through entire sequence

### RNN_head (`policies/models/recurrent_head.py`)

The single entry point that wires together image encoder, transition embedder,
sequence model, and observation embedder. Exposes two methods used by the agent:

- **`forward(actions, rewards, observs, masks)`** — full-trajectory pass for training.
  Inputs are aligned with a dummy step at `t = -1` (mask=0):
  `actions, rewards` are `(T+1, B, dim)` and `observs` is `(T+2, B, dim)` so that
  `observs[t] = o_{t-1}` and `observs[1:]` lines up with `actions, rewards`.
  Returns `joint_embeds` of shape `(T+2, B, embedding_size)` plus a side
  `joint_embeds_target` (only populated when the seq model returns
  `_output_target`, e.g. MATE with `transition_dropout`), and an `info` dict
  logged to WandB.

- **`step(prev_internal_state, prev_action, prev_reward, prev_obs, obs, initial)`** —
  single-step rollout used at eval time (L=1). Updates the seq model's
  internal state and returns `(joint_embed, current_internal_state)`.

**Transition input convention** (see `get_hidden_states`):
- `full_transition=True`: `(o_t, a_t, r_t, o_{t+1} - o_t)` — delta form.
- `full_transition=False, obs_shortcut=True`: `(o_t, a_t, r_t)`.
- `full_transition=False, obs_shortcut=False`: `(a_t, r_t, o_{t+1})`.

**Transition embedder dispatch** (in `__init__`):
- `name == "markov"` → `IdentityModule()` (no memory).
- `name == "mate_rff"` → `RFFEmbedding(...)` (random Fourier features).
- Otherwise → `Mlp(...)` with `**config_seq.embedder.to_dict()` kwargs.

**Dummy step handling**: when `obs_shortcut=True`, the dummy transition at
`t=-1` is dropped before feeding the seq model, and the seq model's
zero-internal-state hidden (`internal_state_to_hidden`) is prepended to the
output. For MATE / MateRff this preserves the learned `init_emb` at `t=-1`;
for other seq models a zero vector is prepended instead. The `h_dummy` at the
top of `forward` then adds an explicit zero at `t=-1` of the final
`(T+2, B, dim)` embedding tensor to align with `observs`.

**Joint embedding**:
- `obs_shortcut=True`: `joint_embed = cat(obs_embed, h_t)`  → `Q(s, h)`
- `obs_shortcut=False`: `joint_embed = h_t`  → `Q(h)`

**`project_output=True`**: L2-normalize both `hidden_states` and
`observs_embeds` and rescale by `sqrt(hidden_dim)` so that each vector has
fixed norm. Use this when the downstream critic is sensitive to drifting
embedding magnitudes (common with MATE because the running-mean output has
no built-in normalization).

**Oracle Markov**: when `seq_model.name == "markov"` and `seq_model.is_oracle`,
the obs wrapper appends a `context_dim` tail (the true latent context) to the
observation. `RNN_head` splits this off via `double_Mlp`: image part →
`image_encoder` (if any) → obs MLP; context tail → context MLP; the two
embeddings are concatenated. The final embedding is
`cat(obs_embed, context_embed, h_t)` (with `hidden_dim` set to 0 since the
seq model is identity).

**Image observations** (`config_seq.image_encoder` set): `_encode_obs` runs the
CNN before the transition is built. For oracle Markov + image, the context
tail bypasses the CNN and is re-concatenated to the encoded image.

**`torch.compile`**: enabled when CUDA is available *and*
`config_seq.compile=True`. Compiles `seq_model`, `observ_embedder`,
`transition_embedder`, and `image_encoder` independently. Disable when
debugging shape/dtype issues — error messages from compiled graphs are noisy.

### MATE Model (`policies/seq_models/mate_vanilla.py`)

Core innovation: instead of attention, maintains a running normalized sum of embeddings.
Internal state is `(cumsum, t)` — a pair of tensors `(hidden, time_count)` of shapes
`(1, B, hidden_size)` and `(1, B, 1)`. The cumulative sum and time count are
both accumulated using `torch.cat([init, x]).cumsum(0)[1:]` rather than
`init + x.cumsum(0)` to avoid an Inductor SplitScan + broadcast crash
(pytorch/pytorch#180221).

```python
# inside forward(inputs, h_0=(hidden, t)):
z = self.embedder(inputs)                              # (T, B, hidden_size)
cumsum     = cat([hidden, z * w], dim=0).cumsum(0)[1:] # (T, B, hidden_size)
t_expanded = cat([t,      w    ], dim=0).cumsum(0)[1:] # (T, B, 1)
output     = cumsum / t_expanded.clamp(min=1e-6)       # running mean
```

The embedder is a `gpt_like_Mlp` (residual MLP with GELU; output LayerNorm
gated by `use_output_ln`, default True).

**`init_emb`** (`(hidden_size,)`): the value placed at `t=-1` so the running
mean is well-defined at `t=0`. Learnable `nn.Parameter` by default;
`init_emb_zero=True` registers it as a zero buffer. `get_zero_internal_state`
returns `(init_emb_expanded, ones)` so the initial transition is counted as 1.

**Gating (`use_gate=True`)** — a per-step scalar `w` controls how much each
embedding contributes to the running mean:
```
w = _GATE_MIN + (1 - 2*_GATE_MIN) * sigmoid(gate(inputs) + noise)
```
where `_GATE_MIN = 0.01` clamps `w ∈ [0.01, 0.99]` to prevent collapse, and
`gate_noise_std` adds optional Gaussian noise to the pre-sigmoid logits
during training. Logs `gates_mean`, `gates_std`, and `gates_collapse_ratio`
(fraction of `raw_w` values below `_GATE_MIN`) per timestep to WandB.
With `use_gate=False` (default), `w=1` everywhere.

**Transition / rollout dropout** — stochastically zero out per-step
contributions to the running sum and time count:
- `transition_dropout` (training only): drops `z * w` and `w` jointly with
  prob `transition_dropout`. The kept mask is sampled per `(T, B, 1)` cell.
  In this mode, MATE additionally returns `_output_target` in `info`
  (computed as if the dropped transitions had been kept) — `RNN_head`
  consumes this as a target for an auxiliary loss.
- `rollout_dropout` (eval only, when `_rollout_dropout_active=True`): same
  mechanism applied at rollout time to study robustness to missing
  transitions.

Both are exposed as floats in `[0, 1)`; assertions enforce the range.

### MateRff Model (`policies/seq_models/mate_rff.py` + `Rff_embedding.py`)

Variant of MATE where the *transition embedder* is a Random Fourier Feature
(RFF) projection instead of an MLP, and the seq model itself is a plain
running mean (no learned embedder, no gating, no dropout). The motivation:
mean-pooled RFF embeddings approximate the MMD between transition
distributions, giving MATE a kernel-method interpretation.

**Wiring** (see `RNN_head.__init__`):
- `transition_embedder = RFFEmbedding(input_dim=transition_size,
  embedding_dim=hidden_dim, kernel=cfg.kernel, normalize_inputs=...)`.
- `seq_model = MateRff(...)` — strips the embedder out of MATE: `z = inputs`
  directly, `w = 1`, no gating, no dropout. Just a running mean of the RFF
  features.
- Asserts `input_size == hidden_size` (RFF output dim must equal the
  hidden dim consumed downstream).

**RFFEmbedding** (`policies/seq_models/Rff_embedding.py`) implements the
cos&sin RFF estimator (Sutherland & Schneider, UAI 2015) over five kernels:

| kernel     | spectral measure                       | notes                                                |
|------------|----------------------------------------|------------------------------------------------------|
| `gaussian` | `N(0, σ⁻² I)`                          | default; PD; bandwidth defaults to `sqrt(input_dim)` |
| `laplace`  | product Cauchy(0, 1/σ)                 | PD; l1-Laplace (not l2)                              |
| `matern`   | Student-t with `df = 2·matern_nu`      | PD; `nu=0.5` reduces to l2-Laplace                   |
| `train`    | Gaussian-initialized, then learned     | `omega` is an `nn.Parameter`; no MMD interpretation  |
| `riesz`    | importance-sampled sliced (Hertrich+24)| CPD; uses regularization `riesz_eps`                 |

For PD kernels, importance weights `sqrt_w` are all 1 and `omega` is a frozen
buffer. The Riesz path uses `_sample_sliced_riesz` (uniform direction on the
sphere × Cauchy 1D frequency) with importance weights
`1/(ξ² + riesz_eps²)`. The pairwise identity `E[z(x)·z(y)] = K(x, y)` only
holds for PD kernels; Riesz instead satisfies the multiset-distance identity
`E[||z̄(μ) - z̄(ν)||²] ∝ MMD²(μ, ν)`, which is the property MATE consumes.

**Forward**:
```python
x        = in_norm(x)                                      # optional InputNorm
proj     = x @ omega.T                                     # (..., num_freq)
out      = sqrt(2) * interleave(sqrt_w · cos(proj), sqrt_w · sin(proj))
```
- `embedding_dim` must be even (cos+sin per frequency).
- The canonical `1/sqrt(2D)` prefactor is dropped to keep activations in
  `[-1, 1]`; the pairwise identity then holds up to a `2D` constant.
- `normalize_inputs=True` wraps inputs in `InputNorm` (running mean/var
  updated only during training). The `mask` argument (when passed by
  `RNN_head`) excludes padded steps from the running stats but does NOT
  zero the forward output — downstream code must mask padded entries
  before pooling. Because frequencies are frozen, early-training drift in
  InputNorm shifts the effective kernel scale.

### MATE / MateRff comparison cheat sheet

|                            | MATE (`mate`)              | MateRff (`mate_rff`)                       |
|----------------------------|----------------------------|--------------------------------------------|
| Transition embedder        | MLP                        | `RFFEmbedding` (frozen by default)         |
| Embedder inside seq model  | `gpt_like_Mlp`             | Identity (`z = inputs`)                    |
| Per-step weight `w`        | learned gate or `1`        | always `1`                                 |
| Dropout                    | transition + rollout       | none                                       |
| `_output_target`           | when `transition_dropout>0`| never                                      |
| Internal state             | `(cumsum, t)`              | `(cumsum, t)` (same shape)                 |
| Interpretation             | learned running mean       | MMD-style kernel mean embedding            |

### Adding a New Sequence Model

Implement a `nn.Module` with:
- `name`: class attribute (string key for registry)
- `hidden_size`: instance attribute (used by `RNN_head` to decide whether to log hidden-norm stats; set to 0 for no-memory models)
- `forward(inputs, h_0, **kwargs) → (output, h_n, info)`: `info` is a dict (may be empty) logged to WandB. May include `_output_target` for auxiliary loss targets — `RNN_head` will pop it before logging.
- `get_zero_internal_state(batch_size, **kwargs) → h_0`
- `internal_state_to_hidden(internal_state) → tensor`: extracts the `(1, B, hidden_size)` hidden tensor. Only called when `obs_shortcut=True` and `name in ("mate", "mate_rff")`; other models get a zero-vector dummy hidden prepended instead.

Optionally accept `obs_emb=...` kwarg in `forward` (some models like SplAgger consume it). Register in `SEQ_MODELS` dict in `policies/seq_models/__init__.py`.

### Registries

Components are looked up by string name using registries:
- `SEQ_MODELS` in `policies/seq_models/__init__.py` — maps name → class
- `RL_ALGORITHMS` in `policies/rl/__init__.py` — maps name → class
- `AGENT_CLASSES` in `policies/models/__init__.py` — maps algo → agent class

### Replay Buffer Layout

Shape: `(T+1, num_episodes, dim)` where `T = max_episode_len`.  
Includes a dummy step at `t=-1` (mask=0) for alignment with the seq model's "previous transition" convention. Buffer is circular — oldest episodes overwritten when full.

## Logging

Training uses **Weights & Biases**. The WandB project name is the registered env string (e.g., `tmaze_passive_T-100`), not `run_name`. Run name is `{env_type}/{env_name}/{run_name}_{timestamp}`.

Checkpoints saved to:
```
logs/{env_type}/{env_name}/{run_name}_{timestamp}/
├── policy_checkpoint_latest.pth
└── buffer_checkpoint_latest.pth  (if normalize_transitions=True)
```

Training logs per-timestep tensors (e.g., gate stats, hidden state norms) as matplotlib figures to WandB under `visualizations/` at `visualize_every * log_interval` intervals.

## Notes

- Models are compiled with `torch.compile` when CUDA is available. Disable for debugging by removing the compile calls in `RNN_head.__init__`.
- `n_env` parallel environments run simultaneously; `log_interval`, `eval_interval`, and `eval_episodes` must all be divisible by `n_env`.
