#!/usr/bin/env bash 
device=1
python main.py --config_env configs/envs/mujoco.py --config_env.env_name ant-dir --config_rl configs/rl/sac_default.py --train_episodes 200000 --config_seq configs/seq_models/lstm_default.py --device ${device} --run_name lstm_1 --seed 1
python main.py --config_env configs/envs/mujoco.py --config_env.env_name ant-dir --config_rl configs/rl/sac_default.py --train_episodes 200000 --config_seq configs/seq_models/markov_default.py --device ${device} --run_name markov_1 --seed 1
python main.py --config_env configs/envs/mujoco.py --config_env.env_name ant-dir --config_rl configs/rl/sac_default.py --train_episodes 200000 --config_seq configs/seq_models/gpt_default.py --device ${device} --run_name gpt_1 --seed 1
done