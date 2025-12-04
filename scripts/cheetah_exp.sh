#!/usr/bin/env bash 
seed=0
python main.py --config_env configs/envs/mujoco.py --config_env.env_name cheetah-vel --config_rl configs/rl/sac_default.py --train_episodes 25000 --config_seq configs/seq_models/hist_default.py --config_seq.seq_model.init_emb_mode parameter --device 1 --run_name hist_${seed} --seed ${seed}
python main.py --config_env configs/envs/mujoco.py --config_env.env_name cheetah-vel --config_rl configs/rl/sac_default.py --train_episodes 25000 --config_seq configs/seq_models/gpt_default.py --device 1 --run_name gpt_${seed} --seed ${seed}
python main.py --config_env configs/envs/mujoco.py --config_env.env_name cheetah-vel --config_rl configs/rl/sac_default.py --train_episodes 25000 --config_seq configs/seq_models/lstm_default.py --device 1 --run_name lstm_${seed} --seed ${seed}
python main.py --config_env configs/envs/mujoco.py --config_env.env_name cheetah-vel --config_rl configs/rl/sac_default.py --train_episodes 25000 --config_seq configs/seq_models/markov_default.py --device 1 --run_name markov_${seed} --seed ${seed}
python main.py --config_env configs/envs/mujoco.py --config_env.env_name cheetah-vel --config_rl configs/rl/sac_default.py --train_episodes 25000 --config_seq configs/seq_models/markov_default.py --config_seq.seq_model.is_oracle True --device 1 --run_name oracle_${seed} --seed ${seed}
done