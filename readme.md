# MATE: Solving Contextual Markov Decision Processes with Memory of Accumulated Transition Embeddings

## About this Repository
This repository is an experimental testbed for the paper **"MATE: Solving Contextual Markov Decision Processes with Memory of Accumulated Transition Embeddings"**. 

**Scope & Framework:** 
This work falls under the **Memory-based RL framework** for solving Contextual MDPs. In this framework, a sequence of transitions `(s, a, r, s')` is processed by a sequence model, and the result is used as memory for decision-making.

**MATE Architecture:**
Unlike traditional Transformers or RNNs, MATE utilizes the **summation** of transition embeddings as its memory. To prevent the memory from expanding indefinitely, an `init_emb` is added, and the result is either divided by the number of transitions or bounded via **hyperspherical projection**.

**Environments & Baselines:**
This repository provides environments to test Memory-based RL combining:
- **Memory Architectures:** MATE, SplAgger, Transformer (GPT), RNN/LSTM/GRU, Markov
- **Contextual MDP Environments:** Tmaze, Mujoco, Metaworld

---

## Modular Design
The code has a modular design which requires *three* configuration files. We hope that such design could facilitate future research on different environments, RL algorithms, and sequence models.

- `config_env`: specify the environment, with `config_env.env_name` specifying the exact (memory / credit assignment) length of the task
    - Passive T-Maze (this work)
    - Active T-Maze (this work)
    - Passive Visual Match (based on [Hung et al., 2018])
    - Key-to-Door (based on [Raposo et al., 2021])
- `config_rl`: specify the RL algorithm and its hyperparameters
    - DQN (with epsilon greedy)
    - Continuous SAC (we find `--freeze_critic` can prevent degradation in the actor update)
- `config_seq`: specify the memory architecture and its hyperparameters
    - MATE and MATE+MSC
    - SplAgger
    - RNN, LSTM, and GRU
    - Transformer (GPT-2) [Radford et al., 2019]
    - Markov and oracle Markov baselines

### Core training structure

- `main.py` resolves the three configs, creates non-autoresetting vector
  environments, initializes W&B, and starts `Learner`.
- `policies/learner.py` coordinates rollout collection, replay updates,
  evaluation, logging, and checkpointing.
- `policies/models/policy_rnn_dqn.py` and `policy_rnn_sac.py` each own their
  complete RL algorithm state and update logic.
- `policies/models/recurrent_head.py` is the shared observation/transition
  encoder and sequence-memory assembly point.
- `buffers/rollout_buffer.py` stores full episodes and optionally samples
  independent contiguous truncated-BPTT windows.

## Installation
We use python 3.10 and list the requirements in [`requirements.txt`](https://github.com/twni2016/Memory-RL/blob/main/requirements.txt). 
```bash
conda create -y -n mate python=3.10
conda activate mate
pip install -r requirements.txt
```

### Symbolic Alchemy setup (optional)
The Symbolic Alchemy environment additionally needs [`dm_alchemy`](https://github.com/google-deepmind/dm_alchemy), which is **not on PyPI** (the repo is archived and GitHub-only) — so `pip install dm_alchemy` fails. It is intentionally kept out of `requirements.txt`. Install it into the activated `mate` env with the helper script (symbolic Alchemy needs no Docker/Unity/GL, unlike the 3D version):
```bash
bash scripts/install_dm_alchemy.sh mate
```
The script installs the DeepMind deps as wheels, clones the archived source, compiles its protobufs, and exposes the package via a `.pth` file (its own `setup.py` is broken on modern `setuptools`, which dropped `pkg_resources`). It leaves `numpy`/`scipy`/`torch` untouched and runs an import smoke test at the end. If you hit a protobuf `"Descriptors cannot be created directly"` error at runtime, prepend `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` to your command.


### CARL Vehicle Racing setup (optional)
CARL Vehicle Racing wraps Gymnasium's Box2D `CarRacing`, which needs `Box2D` and `pygame`. They are intentionally kept out of `requirements.txt` (like `dm_alchemy`) since every other environment runs without them. `box2d-py` builds from source, so install `swig` first:
```bash
conda install -y -c conda-forge swig
pip install "gymnasium[box2d]" pygame
```
No display or extra environment variable is needed: CarRacing renders to offscreen `pygame.Surface`es, so both the observations and `visualize_env=True` eval videos work headless with no `DISPLAY`. (`SDL_VIDEODRIVER=dummy` is only relevant for `render_mode="human"`, which this repo never uses. `MUJOCO_GL` is unrelated — CARL is Box2D, not MuJoCo.)
Run the image encoder with `torch.compile` disabled — `--config_seq.use_image_encoder=True --config_seq.compile=False`. The CNN loss graph currently breaks compilation two ways: Triton 3.4 can fail codegen outright (`PassManager::run failed`), and `seq_model.use_ema_init_emb=True` updates `init_emb` in place during forward, which the compiled backward rejects. Only the loss graph is compiled, so disabling it costs little.

### Mamba setup (optional)
The Mamba baseline (`configs/seq_models/mamba_default.py`) needs `mamba_ssm` + `causal_conv1d` (CUDA GPU only), kept out of `requirements.txt`. After installing the requirements:
```bash
bash scripts/install_mamba.sh mate
```
It installs the prebuilt wheels matching your torch/CUDA/python (torch 2.6–2.10). If they can't load (e.g. `GLIBC_2.32 not found` on Ubuntu 20.04), it builds from source instead: this takes 30+ min and needs `nvcc` with the same CUDA major version as torch (`export CUDA_HOME=...`). It prints `mamba OK` on success.

## Setting Environment Variables (For MuJoCo Experiments Visualization)
The MuJoCo simulator renders images using OpenGL and supports three different backends: glfw, egl, and osmesa. You can choose the appropriate backend by setting the MUJOCO_GL environment variable.

When rendering with a Window System on GPU, run:
```
export MUJOCO_GL=glfw
```
When rendering headless on GPU, run:
```
export MUJOCO_GL=egl
```
When rendering headless on CPU, run:
```
export MUJOCO_GL=osmesa
```
To avoid manually setting the environment variable every time you start your experiments, you can add the appropriate export command to your shell's startup file (`~/.bashrc`).


## Experiments

To run T-Maze passive with a corridor length of 100 with Mate-based agent:
```bash
python main.py --config_env configs/envs/tmaze_passive.py --config_env.env_name 100 --config_rl configs/rl/dqn_default.py --train_episodes 20000 --config_seq configs/seq_models/mate_default.py --device 0 --run_name test
```
You can adjust the corridor length by setting --config_env.env_name. For T-Maze active experiment, replace --config_env configs/envs/tmaze_passive.py with --config_env configs/envs/tmaze_active.py.

To run the same experiment with Transformer-based or LSTM-based agent, set --config_seq to configs/seq_models/gpt_default.py or configs/seq_models/lstm_default.py

To run mujoco benchmark experiment for cheetah-vel environment with Transformer-based agent:
```bash
python main.py --config_env configs/envs/mujoco.py --config_env.env_name cheetah-vel --config_rl configs/rl/sac_default.py --train_episodes 25000 --config_seq configs/seq_models/gpt_default.py --device 0 --run_name test
```
To run the other mujoco environments, set --config_env.env_name to one of ["cheetah-vel", "ant-dir", "hopper-param", "walker-param"]

To run metaworld benchmark experiment for ML10 environment with LSTM-based agent:
```bash
python main.py --config_env configs/envs/metaworld.py --config_env.env_name ML10 --config_rl configs/rl/sac_default.py --train_episodes 25000 --config_seq configs/seq_models/lstm_default.py --device 0 --run_name test
```
To run the experiment on ML45 environment, set --config_env.env_name to ML45

For pixel-based environments such as CARL Vehicle Racing, enable the image encoder with `--config_seq.use_image_encoder=True` and disable `torch.compile` (see the CARL setup section above). The context is (vehicle, track): `config_env.num_tracks` (default 10; `<= 0` = a fresh track per episode) fixes the track set, and `config_env.frame_stack` (default 2) stacks two frames.
```bash
python main.py --config_env configs/envs/carl_vehicle_racing.py --config_env.env_name all --config_rl configs/rl/sac_default.py --config_seq configs/seq_models/mate_default.py --config_seq.use_image_encoder=True --config_seq.compile=False --train_episodes 10000 --device 0 --run_name test
```
The CNN settings (`channels`, `kernel_sizes`, `strides`, `embedding_size`) can be overridden via e.g. `--config_seq.image_encoder.embedding_size=64`. `image_shape` is not one of them — it is read from the env at startup so that `config_env.frame_stack` and the CNN can never disagree.

By default, the logging data is stored in `logs/` folder.  You can visualize the training log using Weights & Biases (WANDB).

Current checkpoints are written as:

```text
training_checkpoint.pth
buffer_checkpoint.pth   # skipped when --save_buffer=False
```

The training checkpoint format is versioned. Checkpoints created before the
core agent refactor are not supported.

## Regression tests

The core shape, replay, agent-state, config, and checkpoint contracts use the
standard-library test runner:

```bash
python -m unittest discover -s tests -p 'test_*.py'
```

## Acknowledgement

The code is largely based on prior works:
- [POMDP Baselines](https://github.com/twni2016/pomdp-baselines)
- [Hugging Face Transformers](https://github.com/huggingface/transformers)
