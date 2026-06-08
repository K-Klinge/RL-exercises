#!/bin/zsh

python3.11 -m rl_exercises.week_6.actor_critic -m \
                  env.name=LunarLander-v3 \
                  seed=0,1,2,3,4,5,6,7,8,9 \
                  hydra.sweep.dir=outputs/ppo_sweep \
                  hydra.sweep.subdir='${env.name}/actor_critic/seed_${seed}' \
                  hydra/launcher=joblib \
                  hydra.launcher.n_jobs=10

python3.11 -m rl_exercises.week_6.ppo -m \
                  env.name=LunarLander-v3 \
                  agent.use_value_clipping=False \
                  agent.use_early_stopping=False \
                  seed=0,1,2,3,4,5,6,7,8,9 \
                  hydra.sweep.dir=outputs/ppo_sweep \
                  hydra.sweep.subdir='${env.name}/vanilla/seed_${seed}' \
                  hydra/launcher=joblib \
                  hydra.launcher.n_jobs=10

python3.11 -m rl_exercises.week_6.ppo -m \
            env.name=LunarLander-v3 \
            agent.use_value_clipping=True \
            agent.use_early_stopping=True \
            seed=0,1,2,3,4,5,6,7,8,9 \
            hydra.sweep.dir=outputs/ppo_sweep \
            hydra.sweep.subdir='${env.name}/improved/seed_${seed}' \
            hydra/launcher=joblib \
            hydra.launcher.n_jobs=10