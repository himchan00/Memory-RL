# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Memory-RL is an experimental testbed for **MATE** (Memory of Accumulated Transition Embeddings), a memory-based RL framework for solving Contextual MDPs (POMDPs). Unlike Transformers or RNNs, MATE maintains memory by summing transition embeddings — preventing unbounded memory growth while enabling context-sensitive behavior.

**Supported environments:** T-Maze (passive/active/detour), MuJoCo (cheetah-vel, ant-dir, hopper-param, walker-param), Metaworld (ML10, ML45)  
**Supported algorithms:** DQN (discrete), SAC (continuous)  
**Memory architectures:** MATE, GPT-2, LSTM, GRU, RNN, Markov (no memory)

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
    │   ├── transition_embedder: MLP(prev_obs, action, reward, obs) → hidden_dim
    │   ├── seq_model: MATE/GPT2/LSTM/Markov processes embedded transitions
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

### MATE Model (`policies/seq_models/mate_vanilla.py`)

Core innovation: instead of attention, maintains a running normalized sum of embeddings:
```python
cumsum = hidden + (z * w).cumsum(dim=0)        # accumulate weighted embeddings
output = (cumsum + self.init_emb) / t_expanded # normalize by accumulated weight
```
Internal state is `(cumsum, t)` — not a fixed-size hidden vector. The embedder inside MATE is a `gpt_like_Mlp` (residual MLP with GELU, no output LayerNorm).

With `use_gate=False` (default): `w=1` (simple sum). With `use_gate=True`: `w = sigmoid(gate(inputs))` — a per-step scalar weight; logs `gates_mean` and `gates_std` per timestep to WandB.

### Adding a New Sequence Model

Implement a `nn.Module` with:
- `name`: class attribute (string key for registry)
- `forward(inputs, h_0) → (output, h_n, info)`: `info` is a dict (may be empty) logged to WandB
- `get_zero_internal_state(batch_size, **kwargs) → h_0`
- `internal_state_to_hidden(internal_state) → tensor`: extracts the hidden state tensor

Register in `SEQ_MODELS` dict in `policies/seq_models/__init__.py`.

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
