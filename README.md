## Olympics Running Competition

> 郭奕玮、郭子奥、朱慕之、刘鹤洋

Modified from [https://github.com/jidiai/Competition_Olympics-Running](https://github.com/jidiai/Competition_Olympics-Running).

---
## Prerequisites

The same as the officially provided `requirements.txt`.

---
## Testing the trained model

For **ED-SP-Spos** (the best model in our paper), it is stored in `rl_trainer/models/olympics-running/ppo/shuffle_use_dist_self_play_shuffle_pos`.
So you can test it with 
```shell
python evaluate.py --my_ai \
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
    --episode 20 \
    --num_frame \
    1 > results.txt
```
Note that this is against baseline trained on shuffled maps.

We also provide ED-SP-Spos trained on single maps. They are in `.../ppo/map*_use_dist_self_play_shuffle_pos`. You can test them with similar commands.
