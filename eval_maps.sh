#!/bin/bash

# baseline
#for map in $(seq 11); do
#  python evaluation.py --my_ai \
#    ppo \
#    --my_ai_run_dir \
#    baseline_map"$map" \
#    --my_ai_run_episode \
#    1500 \
#    --opponent \
#    random \
#    --map \
#    "$map" \
#    --actor_hidden_layers \
#    1 \
#    --critic_hidden_layers \
#    1 \
#    --episode 100 \
#    --num_frame \
#    1 > rl_trainer/models/olympics-running/ppo/baseline_map"$map"/results.txt &
#done

# use episode dist only
#for map in $(seq 11); do
#    python evaluation.py --my_ai \
#    ppo \
#    --my_ai_run_dir \
#    map"$map"_use_dist \
#    --my_ai_run_episode \
#    1500 \
#    --opponent \
#    ppo \
#    --opponent_run_dir \
#    baseline_map"$map" \
#    --opponent_run_episode \
#    1500 \
#    --map \
#    "$map" \
#    --actor_hidden_layers \
#    1 \
#    --critic_hidden_layers \
#    1 \
#    --episode 100 \
#    --num_frame \
#    1 > rl_trainer/models/olympics-running/ppo/map"$map"_use_dist/results.txt &
#done

# use step dist only
#for map in $(seq 11); do
#    python evaluation.py --my_ai \
#    ppo \
#    --my_ai_run_dir \
#    map"$map"_use_step_dist \
#    --my_ai_run_episode \
#    1500 \
#    --opponent \
#    ppo \
#    --opponent_run_dir \
#    baseline_map"$map" \
#    --opponent_run_episode \
#    1500 \
#    --map \
#    "$map" \
#    --actor_hidden_layers \
#    1 \
#    --critic_hidden_layers \
#    1 \
#    --episode 100 \
#    --num_frame \
#    1 > rl_trainer/models/olympics-running/ppo/map"$map"_use_step_dist/results.txt &
#done


# use step dist + summed hit wall penalty
#for map in $(seq 11); do
#    python evaluation.py --my_ai \
#    ppo \
#    --my_ai_run_dir \
#    map"$map"_use_step_dist_summed_hit_wall \
#    --my_ai_run_episode \
#    1500 \
#    --opponent \
#    ppo \
#    --opponent_run_dir \
#    baseline_map"$map" \
#    --opponent_run_episode \
#    1500 \
#    --map \
#    "$map" \
#    --actor_hidden_layers \
#    1 \
#    --critic_hidden_layers \
#    1 \
#    --episode 100 \
#    --num_frame \
#    1 > rl_trainer/models/olympics-running/ppo/map"$map"_use_step_dist_summed_hit_wall/results.txt &
#done

# use episode dist + self play
#for map in $(seq 11); do
#    python evaluation.py --my_ai \
#    ppo \
#    --my_ai_run_dir \
#    map"$map"_use_dist_self_play \
#    --my_ai_run_episode \
#    2000 \
#    --opponent \
#    ppo \
#    --opponent_run_dir \
#    baseline_map"$map" \
#    --opponent_run_episode \
#    1500 \
#    --map \
#    "$map" \
#    --actor_hidden_layers \
#    1 \
#    --critic_hidden_layers \
#    1 \
#    --episode 100 \
#    --num_frame \
#    1 > rl_trainer/models/olympics-running/ppo/map"$map"_use_dist_self_play/results.txt &
#done

# use episode dist + self play + shuffle pos
#for map in $(seq 11); do
#    python evaluation.py --my_ai \
#    ppo \
#    --my_ai_run_dir \
#    map"$map"_use_dist_self_play_shuffle_pos \
#    --my_ai_run_episode \
#    2000 \
#    --opponent \
#    ppo \
#    --opponent_run_dir \
#    baseline_map"$map" \
#    --opponent_run_episode \
#    1500 \
#    --map \
#    "$map" \
#    --actor_hidden_layers \
#    1 \
#    --critic_hidden_layers \
#    1 \
#    --episode 100 \
#    --num_frame \
#    1 > rl_trainer/models/olympics-running/ppo/map"$map"_use_dist_self_play_shuffle_pos/results.txt &
#done

python evaluation.py --my_ai \
    ppo \
    --my_ai_run_dir \
    shuffle_use_dist_self_play_shuffle_pos \
    --my_ai_run_episode \
    2000 \
    --opponent \
    ppo \
    --opponent_run_dir \
    shuffle_map \
    --opponent_run_episode \
    1500 \
    --map \
    all \
    --actor_hidden_layers \
    1 \
    --critic_hidden_layers \
    1 \
    --episode 100 \
    --num_frame \
    1 > rl_trainer/models/olympics-running/ppo/shuffle_use_dist_self_play_shuffle_pos/results.txt

#wait
echo "ALL DONE"
