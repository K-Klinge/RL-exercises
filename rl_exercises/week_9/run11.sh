#!/bin/zsh

# Level 1.1
python3.11 -m rl_exercises.week_9.dyna_ppo -m \
              seed=0,1,2,3,4,5,6,7,8,9 \
              train.total_steps=15000 \
              train.eval_interval=1000 \
              agent.use_model=True,False \
              hydra.sweep.dir=outputs/ppo_sweep_11 \
              hydra.sweep.subdir='${env.name}/${agent.use_model}/seed_${seed}' \
              hydra/launcher=joblib \
              hydra.launcher.n_jobs=10
