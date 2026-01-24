#!/usr/bin/env bash 
device=0
python main.py --config_env configs/envs/mujoco.py --config_env.env_name walker-param --config_rl configs/rl/sac_default.py --train_episodes 150000 --config_seq configs/seq_models/gpt_default.py --device ${device} --run_name gpt_0 --seed 0
python main.py --config_env configs/envs/mujoco.py --config_env.env_name walker-param --config_rl configs/rl/sac_default.py --train_episodes 150000 --config_seq configs/seq_models/lstm_default.py --device ${device} --run_name lstm_0 --seed 0
python main.py --config_env configs/envs/mujoco.py --config_env.env_name walker-param --config_rl configs/rl/sac_default.py --train_episodes 150000 --config_seq configs/seq_models/markov_default.py --device ${device} --run_name markov_0 --seed 0
done