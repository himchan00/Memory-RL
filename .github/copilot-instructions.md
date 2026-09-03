# Memory-RL Copilot Instructions

## Project and commands

Memory-RL is a Python 3.10 research codebase for contextual/POMDP reinforcement
learning. It combines environment configs, DQN or SAC, and a sequence-memory
model such as MATE, SplAgger, GPT-2, LSTM/GRU/RNN, or a Markov baseline.

Set up the main environment with:

```bash
conda create -y -n mate python=3.10
conda activate mate
pip install -r requirements.txt
```

Symbolic Alchemy additionally requires the archived, non-PyPI `dm_alchemy`
package:

```bash
bash scripts/install_dm_alchemy.sh mate
```

Run training by supplying all three config files:

```bash
python main.py \
  --config_env=configs/envs/tmaze_passive.py \
  --config_env.env_name=100 \
  --config_rl=configs/rl/dqn_default.py \
  --config_seq=configs/seq_models/mate_default.py \
  --train_episodes=20000 \
  --device=0 \
  --run_name=tmaze_mate
```

The documented regression-test runner is the standard library `unittest`
runner, not pytest:

```bash
# Full suite
python -m unittest discover -s tests -p 'test_*.py'

# One test method
python -m unittest tests.test_MODULE.TestCaseName.test_method_name
```

For MuJoCo rendering, set `MUJOCO_GL=glfw` for a local window,
`MUJOCO_GL=egl` for headless GPU rendering, or `MUJOCO_GL=osmesa` for
headless CPU rendering.

## Architecture

The runtime is assembled in this order:

```text
main.py
  -> config_env.create_fn registers the Gymnasium environment
  -> config_seq/config_rl update_fn values are finalized from episode length
  -> non-autoresetting AsyncVectorEnv instances are created
  -> Learner coordinates rollout, replay updates, evaluation, W&B, checkpoints
     -> RolloutBuffer stores time-major full episodes or sampled BPTT windows
     -> AGENT_CLASSES selects the DQN or SAC recurrent agent
        -> RNN_head encodes observations and transitions
           -> SEQ_MODELS selects MATE/GPT/RNN/SplAgger/Markov
           -> conditioner combines current observation and memory
        -> the agent owns its algorithm heads, losses, optimizers, and targets
```

`policies/models/policy_rnn_dqn.py` and `policy_rnn_sac.py` each own the
complete algorithm state. There is no separate RL-algorithm implementation
layer to update; `AGENT_CLASSES` in `policies/models/__init__.py` is the
dispatch point.

`policies/models/recurrent_head.py` is the shared boundary between replay data
and every sequence model. It owns optional image encoding, observation and
transition `InputNorm`, transition construction, sequence execution, absolute
position encoding, and observation-memory conditioning.

## Configuration conventions

- Every run requires `config_env`, `config_rl`, and `config_seq`.
- Environment configs start from `configs/envs/common.py`, expose
  `create_fn(config) -> (config, registered_env_name)`, register the environment,
  and delete `config.create_fn` before returning.
- RL and sequence configs expose `update_fn`; `utils/experiment.py` invokes
  these only after the actual `max_episode_steps` is known. Update functions
  must delete `config.update_fn`.
- Put shared sequence defaults in `configs/seq_models/common.py`; keep
  `RNN_head` as the sole consumer of top-level `config_seq` architecture flags.
- `log_interval`, `eval_interval`, and `eval_episodes` must each be divisible
  by `config_env.n_env`.
- Use `--config_section.flag=value` syntax for `ml_collections` overrides,
  especially booleans such as `--config_env.visualize_env=False`.
- `budget.py` is outdated and must not be used as an implementation reference.

## Replay and sequence alignment

Replay data is time-major and includes a dummy row at `t=-1`:

- actions, rewards, terminals, and masks: `(T+1, B, dim)`
- `observations` and `next_observations`: explicit `(T+1, B, obs_dim)` rows
- `mask[0]` is always zero; sequence models and losses depend on this alignment
- `transition_t` records the environment transition index for each row

Do not remove or casually shift the dummy row. Keep DQN and SAC batch handling
shared through `policies/models/off_policy_utils.py`.

`RNN_head` constructs transition inputs as:

- `full_transition=True`: `(o_t, a_t, r_t, o_{t+1} - o_t)`
- `full_transition=False` with `obs_shortcut=True`: `(o_t, a_t, r_t)`
- `full_transition=False` with `obs_shortcut=False`: `(a_t, r_t, o_{t+1})`

`max_seq_len` counts real transitions. When it is shorter than an episode,
`RolloutBuffer` samples an independent contiguous window per batch item, adds
the dummy/reset row, resets memory at the window boundary, and preserves
`transition_t` so absolute positional encoding still uses environment time.

## Sequence-model conventions

- Add sequence models to `policies/seq_models/SEQ_MODELS`. A model provides a
  string `name`, `hidden_size`, `forward(inputs, h_0, **kwargs)`,
  `get_zero_internal_state(batch_size, **kwargs)`, and, when needed,
  `internal_state_to_hidden`.
- Sequence-model `forward` returns `(output, next_internal_state, info)`.
  `info` is forwarded to training/W&B logging; `_aux_loss` is the reserved
  differentiable auxiliary-loss channel.
- `RNN_head.transition_embedder` is identity for MATE and Markov. For MATE,
  the raw post-`InputNorm` transition tuple reaches `Mate.embedder`, which
  owns the full transition embedding pipeline: an input projection
  `Linear(transition_size→hidden_dim) → LeakyReLU → Dropout(dropout_emb)`
  followed by exactly `n_layer` hidden-size
  `Linear → LeakyReLU → Dropout(dropout_ff)` blocks. Other non-Markov models
  receive the shared linear transition projection first.
- With `obs_shortcut=True`, the selected conditioner combines encoded current
  observation and memory. The default Markov configuration uses concat
  conditioning with no sequence-memory readout.
- For oracle Markov plus an image encoder, only the image prefix goes through
  the CNN; preserve and reattach the latent-context tail.

Keep tensors placed in a sequence model's `info` dictionary on-device and
detached as appropriate. Do not call `.item()`, `.cpu()`, print a CUDA tensor,
or branch on a scalar CUDA tensor in the training forward path. `Learner`
batches the CPU transfer at logging time.

`config_seq.compile=True` lazily compiles the CUDA training-loss graph only.
Rollout, optimizer/scheduler steps, and target updates remain eager; disable it
while debugging graph shape or dtype failures.

## Environments, logging, and checkpoints

- `envs/make_env.py` creates the registered environment, then applies
  `KEpisodeWrapper` for `k > 1`, then the oracle wrapper.
- Symbolic Alchemy already represents multiple trials in one native
  meta-episode; use `--k=1` and configure `num_trials` instead of nesting the
  k-shot wrapper.
- W&B uses the registered environment name as the project. Local run data is
  placed under `<save_dir>/<env_type>/<env_name>/<run_name>_<timestamp>/`.
- Checkpoints are the versioned `training_checkpoint.pth` plus
  `buffer_checkpoint.pth`. Agents must include model, optimizer, scheduler,
  target, and algorithm-specific state in `training_state_dict()` and restore
  it in `load_training_state_dict()`. Pre-refactor checkpoint formats are not
  supported.

## AMLT jobs

Before editing `amlt/*.yaml`, also follow `amlt/CLAUDE.md`. Common commands are:

```bash
amlt run amlt/<config>.yaml <experiment_name>
amlt status <experiment_name>
amlt logs <experiment_name> :<job_name>
```

AMLT expands YAML commands with Python `string.Template`: escape shell dollars
as `$$` while leaving AMLT substitutions such as `${BASE_DIR}` unchanged.
Singularity jobs do not have sudo; install system libraries with conda. For
headless MuJoCo on the configured cluster, use OSMesa with `mesalib<25.1`,
`MUJOCO_GL=osmesa`, and `PYOPENGL_PLATFORM=osmesa`.
