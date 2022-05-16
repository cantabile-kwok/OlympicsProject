## Olympics Running Competition

Modified from [https://github.com/jidiai/Competition_Olympics-Running](https://github.com/jidiai/Competition_Olympics-Running).
- clean some code used for Jidi competitions
- new pipeline for developing

### Usage

```shell
git clone https://github.com/Leo-xh/Competition_Olympics-Running.git
cd Competition_Olympics-Running

# training ppo with random opponent
python rl_trainer/main.py --device cuda --map 1  # 可以用不同的map，或者用--shuffle_map

# evaluating ppo with random opponent
python evaluation.py --my_ai ppo --my_ai_run_dir run1 --my_ai_run_episode 1500 --map 1  # run1可以改成自己的output directory
```

### Suggestions

1. The random opponent may be too weak for developing new algorithms, you can implement other rule-based agents to compete with your algorithm.
2. You can also consider self-paly based training methods in training your agent.
3. For training a ppo algorithm, the given metrics may not be enough, you can add other metrics, e.g. clipping ratio, to help monitoring the training process.
4. Single-agent PPO may not work in difficult maps, and you should train your agent with `--shuffle_map` flag finally.


