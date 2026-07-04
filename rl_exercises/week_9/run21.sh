#!/bin/zsh

# Level 2.1.a
python3.11 -m rl_exercises.week_9.dyna_ppo -m \
              seed=0,1,2,3,4,5,6,7,8,9 \
              train.total_steps=15000 \
              train.eval_interval=1000 \
              agent.imag_horizon=1,3,5,10,20 \
              hydra.sweep.dir=outputs/ppo_sweep_21a \
              hydra.sweep.subdir='${env.name}/${agent.imag_horizon}/seed_${seed}' \
              hydra/launcher=joblib \
              hydra.launcher.n_jobs=10

# Level 2.1.b
python3.11 -m rl_exercises.week_9.dyna_ppo -m \
              seed=0,1,2,3,4,5,6,7,8,9 \
              agent.model_epochs=1 \
              agent.imag_batches=5 \
              train.total_steps=15000 \
              train.eval_interval=1000 \
              hydra.sweep.dir=outputs/ppo_sweep_21b \
              hydra.sweep.subdir='${env.name}/conservative/seed_${seed}' \
              hydra/launcher=joblib \
              hydra.launcher.n_jobs=10

python3.11 -m rl_exercises.week_9.dyna_ppo -m \
              seed=0,1,2,3,4,5,6,7,8,9 \
              agent.model_epochs=3 \
              agent.imag_batches=10 \
              train.total_steps=15000 \
              train.eval_interval=1000 \
              hydra.sweep.dir=outputs/ppo_sweep_21b \
              hydra.sweep.subdir='${env.name}/balanced/seed_${seed}' \
              hydra/launcher=joblib \
              hydra.launcher.n_jobs=10

python3.11 -m rl_exercises.week_9.dyna_ppo -m \
              seed=0,1,2,3,4,5,6,7,8,9 \
              agent.model_epochs=5 \
              agent.imag_batches=20 \
              train.total_steps=15000 \
              train.eval_interval=1000 \
              hydra.sweep.dir=outputs/ppo_sweep_21b \
              hydra.sweep.subdir='${env.name}/aggressive/seed_${seed}' \
              hydra/launcher=joblib \
              hydra.launcher.n_jobs=10

# Level 2.1.c
python3.11 -m rl_exercises.week_9.dyna_ppo -m \
              seed=0,1,2,3,4,5,6,7,8,9 \
              train.total_steps=15000 \
              train.eval_interval=1000 \
              agent.max_buffer_size=1000,5000,10000,50000 \
              hydra.sweep.dir=outputs/ppo_sweep_21c \
              hydra.sweep.subdir='${env.name}/${agent.max_buffer_size}/seed_${seed}' \
              hydra/launcher=joblib \
              hydra.launcher.n_jobs=10