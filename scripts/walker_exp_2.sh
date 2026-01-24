#!/usr/bin/env bash 
device=2
python main.py --config_env configs/envs/mujoco.py --config_env.env_name walker-param --config_rl configs/rl/sac_default.py --train_episodes 150000 --config_seq configs/seq_models/markov_default.py --device ${device} --run_name markov_2 --seed 2
python main.py --config_env configs/envs/mujoco.py --config_env.env_name walker-param --config_rl configs/rl/sac_default.py --train_episodes 150000 --config_seq configs/seq_models/gpt_default.py --device ${device} --run_name gpt_2 --seed 2
python main.py --config_env configs/envs/mujoco.py --config_env.env_name walker-param --config_rl configs/rl/sac_default.py --train_episodes 150000 --config_seq configs/seq_models/lstm_default.py --device ${device} --run_name lstm_2 --seed 2
done