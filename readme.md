
## Modular Design
The code has a modular design which requires *three* configuration files. We hope that such design could facilitate future research on different environments, RL algorithms, and sequence models.

- `config_env`: specify the environment, with `config_env.env_name` specifying the exact (memory / credit assignment) length of the task
    - Passive T-Maze (this work)
    - Active T-Maze (this work)
    - Passive Visual Match (based on [Hung et al., 2018])
    - Key-to-Door (based on [Raposo et al., 2021])
- `config_rl`: specify the RL algorithm and its hyperparameters
    - DQN (with epsilon greedy)
    - SAC-Discrete (we find `--freeze_critic` can prevent gradient explosion, see the discussion in Appendix C.1 in the latest version of the arXiv paper). 
- `config_seq`: specify the sequence model and its hyperparameters including training sequence length `config_seq.sampled_seq_len` and number of layers `--config_seq.model.seq_model_config.n_layer` 
    - LSTM [Hochreiter and Schmidhuber, 1997]
    - Transformer (GPT-2) [Radford et al., 2019]

## Installation
We use python 3.10 and list the requirements in [`requirements.txt`](https://github.com/twni2016/Memory-RL/blob/main/requirements.txt). 
```bash
conda create -y -n hist python=3.10
conda activate hist
pip install -r requirements.txt
```

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

To run T-Maze detour with a corridor length of 100 with Hist-based agent:
```bash
python main.py --config_env configs/envs/tmaze_detour.py --config_env.env_name 100 --config_rl configs/rl/dqn_default.py --train_episodes 20000 --config_seq configs/seq_models/hist_default.py --device 0 --run_name test
```
You can adjust the corridor length by setting --config_env.env_name

To run the same experiment with Transformer-based or LSTM-based agent, set --config_seq to configs/seq_models/gpt_default.py or configs/seq_models/lstm_default.py

To run mujoco benchmark experiment for cheetah-vel environment with Transformer-based agent:
```bash
python main.py --config_env configs/envs/mujoco.py --config_env.env_name cheetah-vel --config_rl configs/rl/sac_default.py --train_episodes 20000 --config_seq configs/seq_models/gpt_default.py --device 0 --run_name test
```
To run the other mujoco environments, set --config_env.env_name to one of ["cheetah-vel", "ant-dir", "hopper-param", "walker-param"]

To run metaworld benchmark experiment for ML10 environment with LSTM-based agent:
```bash
python main.py --config_env configs/envs/metaworld.py --config_env.env_name ML10 --config_rl configs/rl/sac_default.py --train_episodes 20000 --config_seq configs/seq_models/lstm_default.py --device 0 --run_name test
```
To run the experiment on ML45 environment, set --config_env.env_name to ML45


The `train_episodes` of each task is specified in [`budget.py`](https://github.com/twni2016/Memory-RL/blob/main/budget.py). 

By default, the logging data is stored in `logs/` folder.  You can visualize the training log using Weights & Biases (WANDB).

## Acknowledgement

The code is largely based on prior works:
- [POMDP Baselines](https://github.com/twni2016/pomdp-baselines)
- [Hugging Face Transformers](https://github.com/huggingface/transformers)

