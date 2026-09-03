# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Memory-RL is an experimental testbed for **MATE** (Memory of Accumulated Transition Embeddings), a memory-based RL framework for solving Contextual MDPs (POMDPs). Unlike Transformers or RNNs, MATE maintains memory by summing transition embeddings — preventing unbounded memory growth while enabling context-sensitive behavior.

**Supported environments:** T-Maze (passive/active/detour), MuJoCo (cheetah-vel, ant-dir, hopper-param, walker-param), Metaworld (ML10, ML45)  
**Supported algorithms:** DQN (discrete), SAC (continuous)  
**Memory architectures:** MATE, SplAgger, GPT-2, LSTM, GRU, RNN, Markov (no memory)

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
- `obs_shortcut`: if True, the encoded observation is fed to a `Conditioner` together with the seq model output → joint embedding (see Joint embedding below). If False, joint embedding is just `h_t`.
- `full_transition`: if True, transition input is `(o_t, a_t, r_t, o_{t+1})`; if False, uses only `(o_t, a_t, r_t)` or `(a_t, r_t, o_{t+1})`
- `conditioning`: one of `"concat" | "film" | "hypernet"` — how `encoded_obs` and `h_t` are combined. See **Joint embedding** in the `RNN_head` section.
- `conditioning_n_layer`: number of MODULATED blocks added after a plain input projection. Default `1` → 1 modulated block (`input_proj` + `(Linear → act → FiLM|Hyper)`). `0` → input_proj only (no modulation, baseline MLP). Convention matches `input projection + n_layer post-projection layers` (same as `Mate.embedder`). ConcatConditioner is unaffected by this semantic (cat is at end, plain Linears throughout).
- `normalize_inputs`: if True, `RNN_head` applies running-mean/var `InputNorm` to (a) the encoded obs (before the conditioner) and (b) the assembled transition tuple (before the transition embedder). Stats updated only during training, with the rollout mask excluding padded steps. `RNN_head` owns both norms.

### Component Hierarchy

```
Learner (policies/learner.py)
├── AsyncVectorEnv (n_env parallel envs)
├── RolloutBuffer (buffers/rollout_buffer.py)  — trajectory storage
└── Agent (Policy_DQN_RNN or Policy_SAC_RNN)
    ├── RNN_head (policies/models/recurrent_head.py)  — core architecture
    │   ├── image_encoder: optional CNN, applied to obs (used for pixel envs)
    │   ├── encoded_obs_norm / transition_input_norm: external InputNorm on obs and transition tuple (when normalize_inputs=True)
    │   ├── transition_embedder: input projection (Linear+LeakyReLU+Dropout) for non-MATE, non-Markov sequence models; Identity for MATE (which owns its projection) and Markov (no memory)
    │   ├── seq_model: MATE/SplAgger/GPT2/LSTM/Markov processes embedded transitions
    │   └── conditioner: ConcatConditioner | FiLMConditioner | HyperConditioner (when obs_shortcut=True)
    ├── Critic: MLP over joint_embed = conditioner(encoded_obs, h_t)
    └── RL Algorithm (policies/rl/dqn.py or sac.py)  — loss computation, action selection
```

### Data Flow

**At inference (one step):**
1. `Learner.act()` calls `agent.act(prev_obs, action, reward, obs, internal_state)`
2. `RNN_head.step()` embeds the transition, updates the seq model's internal state
3. Critic/actor selects action from joint embedding `conditioner(encoded_obs, h_t)`

**At training (full trajectory):**
1. Sample batch of full episodes from `RolloutBuffer`
2. `agent.forward()` processes all timesteps through `RNN_head.forward()`
3. RL loss computed over valid steps (using masks); gradients flow through entire sequence

### RNN_head (`policies/models/recurrent_head.py`)

The single entry point that wires together image encoder, transition embedder,
sequence model, and conditioner. Exposes two methods used by the agent:

- **`forward(actions, rewards, observs, masks)`** — full-trajectory pass for training.
  Inputs are aligned with a dummy step at `t = -1` (mask=0):
  `actions, rewards` are `(T+1, B, dim)` and `observs` is `(T+2, B, dim)` so that
  `observs[t] = o_{t-1}` and `observs[1:]` lines up with `actions, rewards`.
  Returns `joint_embeds` of shape `(T+2, B, embedding_size)` and an `info`
  dict logged to WandB.

- **`step(prev_internal_state, prev_action, prev_reward, prev_obs, obs, initial)`** —
  single-step rollout used at eval time (L=1). Updates the seq model's
  internal state and returns `(joint_embed, current_internal_state)`.

**Transition input convention** (see `get_hidden_states`):
- `full_transition=True`: `(o_t, a_t, r_t, o_{t+1} - o_t)` — delta form.
- `full_transition=False, obs_shortcut=True`: `(o_t, a_t, r_t)`.
- `full_transition=False, obs_shortcut=False`: `(a_t, r_t, o_{t+1})`.

**Transition embedder dispatch** (in `__init__`):
- `name in {"markov", "mate"}` → `IdentityModule()` with `seq_input_size = transition_size`. Markov has no memory, so the transition input is ignored. For MATE, the raw post-`InputNorm` transition tuple reaches `Mate.embedder`.
- Otherwise → `nn.Sequential(Linear(transition_size, hidden_dim), LeakyReLU, Dropout(dropout_emb))` with `seq_input_size = hidden_dim`.

**Dummy step handling**: when `obs_shortcut=True`, the dummy transition at
`t=-1` is dropped before feeding the seq model, and the seq model's
zero-internal-state hidden (`internal_state_to_hidden`) is prepended to the
output. For MATE this preserves the learned `init_emb` at `t=-1`;
for other seq models a zero vector is prepended instead. The `h_dummy` at the
top of `forward` then adds an explicit zero at `t=-1` of the final
`(T+2, B, dim)` embedding tensor to align with `observs`.

**Joint embedding** (via `policies/models/conditioning.py`):
- `obs_shortcut=True`: `joint_embed = conditioner(encoded_obs, h_t)` → `Q(s, h)`. The conditioner class is selected by `config_seq.conditioning`:
  - `"concat"` → `ConcatConditioner` — MLP stack on `encoded_obs`, then `cat(out, h_t)`. `out_dim = mlp_out_dim + cond_dim`. For markov, `cond_dim = 0` (no `h_t`) so it reduces to a plain MLP and `out_dim = mlp_out_dim`.
  - `"film"` → `FiLMConditioner` — plain `Linear(in→hidden)` input projection, then `n_layer` blocks of `(Linear → activation → FiLM(·, h_t))`. FiLM `(γ, β)` heads are zero-initialized so the stack starts as identity (Perez+ 2017, arXiv:1709.07871).
  - `"hypernet"` → `HyperConditioner` — plain `Linear(in→hidden)` input projection, then `n_layer` blocks of `(HyperLinear(·, h_t) → activation)`. Initialized with Hyperfan-In (Chang+ 2020, arXiv:2312.08399).
  - All three share `forward(x, c) → joint` and an `.out_dim` attribute. For `film`/`hypernet`, `n_layer = config_seq.conditioning_n_layer` counts MODULATED blocks added after the plain input projection (consistent with MATE's `input projection + n_layer post-projection layers` convention). `RNN_head.embedding_size = conditioner.out_dim`. `film` / `hypernet` require non-markov (asserted in `__init__`).
- `obs_shortcut=False`: `joint_embed = h_t` → `Q(h)` (no conditioner instantiated; `embedding_size = cond_dim`).

**`_encode_obs` & Oracle Markov**: `_encode_obs` runs the CNN when
`config_seq.image_encoder` is set, otherwise passes obs through unchanged.
For oracle Markov (`seq_model.name == "markov"` and `seq_model.is_oracle`),
the obs wrapper appends a `context_dim` tail (the true latent context);
with an image encoder, only the image part is run through the CNN and the
context tail is re-concatenated to the encoded features; without one, the
`(state, context)` vector passes through unchanged. The resulting
`encoded_obs` then flows through `encoded_obs_norm` and the `conditioner`
like any other obs; with markov, `cond_dim = 0` so `ConcatConditioner`
reduces to a plain MLP and `h_t` is dropped from the join. (`film` /
`hypernet` are disallowed for markov by assertion.) **`torch.compile`**:
`config_seq.compile=True` lazily compiles the agent's CUDA training-loss graph.
Rollout, optimizer/scheduler steps, and target updates remain eager. Disable it
when debugging shape/dtype issues because compiled-graph errors are noisy.

### MATE Model (`policies/seq_models/mate_vanilla.py`)

Core innovation: instead of attention, maintains a running normalized sum of embeddings.
Internal state is `(cumsum, count)` — a pair of tensors with shapes
`(1, B, hidden_size)` and `(1, B, 1)`. The cumulative sum uses
`torch.cat([hidden, z]).cumsum(0)[1:]` rather than
`hidden + z.cumsum(0)` to avoid an Inductor SplitScan + broadcast crash
(pytorch/pytorch#180221). Counts are computed directly from the initial count
and the timestep indices, without allocating a per-transition unit tensor.

```python
# inside forward(inputs, h_0=(hidden, initial_count)):
z = self.embed_transitions(inputs)                             # (T, B, hidden_size)
cumsum = cat([hidden, z], dim=0).cumsum(0)[1:]                  # (T, B, hidden_size)
step_counts = arange(1, T + 1, dtype=initial_count.dtype)      # (T,)
counts = initial_count + step_counts.view(T, 1, 1)             # (T, B, 1)
output = cumsum / counts.clamp(min=1e-6)                       # running mean
```

`RNN_head.transition_embedder` is `IdentityModule()` for MATE, so the raw
post-`InputNorm` transition tuple reaches `Mate.embedder`. `Mate.embedder`
owns the full `transition_size → hidden_size` pipeline: an input
projection `Linear(in→h) → LeakyReLU → Dropout(dropout_emb)`, followed by
exactly `n_layer` additional
`Linear(h→h) → LeakyReLU → Dropout(dropout_ff)` blocks.

With `learn_init_emb=True` (the config default), `init_emb` and
`log_init_weight` define a learned initial-memory prior:
`(init_emb + Σz_i) / (exp(log_init_weight) + t)`.
`get_zero_internal_state` starts from that prior sum and count. With
`learn_init_emb=False`, both start at zero and MATE computes the ordinary
mean of observed transition embeddings.

**MSC v2** (`mate_msc_v2_default.py`) uses two equal-size, disjoint random
transition subsets from each episode and applies symmetric bilinear CPC. Each
subset uses MATE's actual mean memory:
`(init_emb + Σ_{i∈S} z_i) / (init_weight + |S|)`. The init prior/count is
detached from MSC; `msc_detach_z` controls only whether MSC gradients reach
the encoder. `mate_msc_ema_v2_default.py` trains the online encoder with CPC
while a frozen EMA encoder supplies the RL/rollout memory.

### MATE depth knob (`n_layer`)

The input projection and all additional layers live inside `Mate.embedder`;
`RNN_head.transition_embedder` is `IdentityModule()` for MATE. `n_layer`
counts hidden-to-hidden MLP blocks added after the input projection:

| `n_layer` | `Mate.embedder` |
|-----------|-----------------|
| `0` | `Linear(in→h) → LeakyReLU → Dropout(dropout_emb)` |
| `≥1` | `Linear(in→h) → LeakyReLU → Dropout(dropout_emb)` followed by `[Linear(h→h) → LeakyReLU → Dropout(dropout_ff)] × n_layer` |

The internal state remains `(cumsum, count)`, with the optional learned
initial-memory prior controlled by `learn_init_emb`.

### Adding a New Sequence Model

Implement a `nn.Module` with:
- `name`: class attribute (string key for registry)
- `hidden_size`: instance attribute (used by `RNN_head` to decide whether to log hidden-norm stats; set to 0 for no-memory models)
- `forward(inputs, h_0, **kwargs) → (output, h_n, info)`: `info` is a dict (may be empty) logged to WandB.
- `get_zero_internal_state(batch_size, **kwargs) → h_0`
- `internal_state_to_hidden(internal_state) → tensor`: extracts the `(1, B, hidden_size)` hidden tensor. Only called when `obs_shortcut=True` and `name == "mate"`; other models get a zero-vector dummy hidden prepended instead.

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
└── buffer_checkpoint_latest.pth
```

Training logs per-timestep tensors (e.g., hidden state norms) as matplotlib figures to WandB under `visualizations/` at `visualize_every * log_interval` intervals.

### Adding metrics to `info` dict — avoid CPU-GPU sync

When you add a scalar/per-step tensor to a `seq_model.forward` `info` dict (or `RNN_head`'s `d_forward`), **keep it on the GPU**. The Learner moves it to CPU in batch at log time; doing so per-step destroys throughput (commit `d710213` "eliminate GPU-CPU sync points" was the original fix).

**Do** (no sync):
```python
info["init_emb_norm"] = self.init_emb.detach().norm()           # 0-dim GPU tensor
info["memory_norm"]   = output.detach().norm(dim=-1).mean(dim=1) # (T,) GPU tensor
```

**Don't** (forces sync every forward):
```python
info["init_emb_norm"] = self.init_emb.detach().norm().item()    # .item() blocks
info["memory_norm"]   = output.detach().norm(dim=-1).mean(dim=1).cpu() # .cpu() blocks
print(f"norm = {tensor}")                                        # implicit .item()
if tensor > 0: ...                                                # implicit .item() on 0-dim
```

Reductions like `.mean(dim=...)`, `.norm()`, `.std()`, `.abs().max()` stay on-device. Only the final CPU transfer (wandb commit) should sync, and that happens once per `log_interval`. Same rule applies to any tensor written into `d_forward` from `RNN_head.forward`.

## Notes

- Agent training-loss graphs are lazily compiled on CUDA when `config_seq.compile=True`; rollout and optimizer-side state updates remain eager.
- `n_env` parallel environments run simultaneously; `log_interval`, `eval_interval`, and `eval_episodes` must all be divisible by `n_env`.
