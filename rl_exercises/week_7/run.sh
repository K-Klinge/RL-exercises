#!/bin/zsh

python3.11 -m rl_exercises.week_7.rnd_dqn -m \
              env.name=LunarLander-v3 \
              seed=0,1,2,3,4,5,6,7,8,9 \
              train.num_frames=100000 \
              hydra.sweep.dir=outputs/sweep \
              hydra.sweep.subdir='${env.name}/RNDDQNAgent/seed_${seed}' \
              hydra/launcher=joblib \
              hydra.launcher.n_jobs=10

python3.11 -m rl_exercises.week_4.dqn_sol -m \
              env.name=LunarLander-v3 \
              seed=0,1,2,3,4,5,6,7,8,9 \
              train.num_frames=100000 \
              hydra.sweep.dir=outputs/sweep \
              hydra.sweep.subdir='${env.name}/DQNAgent/seed_${seed}' \
              hydra/launcher=joblib \
              hydra.launcher.n_jobs=10

python3.11 -m rl_exercises.week_7.rnd_ppo -m \
              env.name=LunarLander-v3 \
              seed=0,1,2,3,4,5,6,7,8,9 \
              train.total_steps=100000 \
              hydra.sweep.dir=outputs/sweep \
              hydra.sweep.subdir='${env.name}/RNDPPOAgent/seed_${seed}' \
              hydra/launcher=joblib \
              hydra.launcher.n_jobs=10

python3.11 -m rl_exercises.week_6.ppo_sol -m \
              env.name=LunarLander-v3 \
              seed=0,1,2,3,4,5,6,7,8,9 \
              train.total_steps=100000 \
              hydra.sweep.dir=outputs/sweep \
              hydra.sweep.subdir='${env.name}/PPOAgent/seed_${seed}' \
              hydra/launcher=joblib \
              hydra.launcher.n_jobs=10