# Policies

`policies/` contains the experiment runtime, RL agents, and sequence-memory
models.

## Structure

- `learner.py`: training/evaluation orchestration, logging, counters, and
  checkpoint coordination.
- `rollout.py`: typed rollout results and episode trajectory accumulation.
- `models/policy_rnn_dqn.py`: DQN agent, epsilon schedule, critic loss, target
  update, optimizer, and training state.
- `models/policy_rnn_sac.py`: continuous SAC agent, actor/critics, entropy
  temperature, target updates, optimizer, and training state.
- `models/recurrent_head.py`: observation encoding, transition construction,
  sequence-model execution, positional encoding, and observation-memory
  conditioning.
- `models/off_policy_utils.py`: stateless time-major batch conversion and
  gradient-clipping helpers shared by DQN and SAC.
- `seq_models/`: MATE, SplAgger, GPT, RNN/LSTM/GRU, and Markov memory models.

The DQN and SAC agents own their algorithm-specific state directly. There is
no separate RL-algorithm registry layer; `models.AGENT_CLASSES` maps the
`config_rl.algo` values `dqn` and `sac` to their agents.

## Data alignment

Replay batches are time-major. Actions, rewards, masks, and terminals have
shape `(T+1, B, dim)` and retain the dummy `t=-1` row. Observations are rebuilt
as `(T+2, B, obs_dim)` before entering `RNN_head`. This alignment is
shared by DQN and SAC through `off_policy_utils.prepare_recurrent_batch`.

## Checkpoints

Each agent implements `training_state_dict()` and
`load_training_state_dict()`, including model, optimizer, scheduler, and
algorithm-specific exploration or temperature state. `Learner` stores this in
the versioned `training_checkpoint.pth`; replay state remains in
`buffer_checkpoint.pth`.
