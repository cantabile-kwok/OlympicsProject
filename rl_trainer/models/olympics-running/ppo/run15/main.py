import argparse
import datetime
import random
import sys
from pathlib import Path
from pprint import pprint
from typing import Dict
from tqdm import tqdm
import shutil
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

base_dir = str(Path(__file__).resolve().parent.parent)
sys.path.append(base_dir)

from collections import deque, namedtuple

from env.chooseenv import make

from rl_trainer.algo.ppo import PPO
from rl_trainer.algo.random import random_agent
from rl_trainer.algo.rule import frozen_agent
from rl_trainer.log_path import *
from rl_trainer.algo.pool import agent_pool

actions_map = {
    0: [-100, -30],
    1: [-100, -18],
    2: [-100, -6],
    3: [-100, 6],
    4: [-100, 18],
    5: [-100, 30],
    6: [-40, -30],
    7: [-40, -18],
    8: [-40, -6],
    9: [-40, 6],
    10: [-40, 18],
    11: [-40, 30],
    12: [20, -30],
    13: [20, -18],
    14: [20, -6],
    15: [20, 6],
    16: [20, 18],
    17: [20, 30],
    18: [80, -30],
    19: [80, -18],
    20: [80, -6],
    21: [80, 6],
    22: [80, 18],
    23: [80, 30],
    24: [140, -30],
    25: [140, -18],
    26: [140, -6],
    27: [140, 6],
    28: [140, 18],
    29: [140, 30],
    30: [200, -30],
    31: [200, -18],
    32: [200, -6],
    33: [200, 6],
    34: [200, 18],
    35: [200, 30],
}  # dicretise action space

algo_name_list = ["ppo"]
algo_list = [PPO]
algo_map = dict(zip(algo_name_list, algo_list))


def get_game(seed: int = None, config: Dict = None):
    return make("olympics-running", seed, config)


def setup_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--game_name", default="olympics-running", type=str)
    parser.add_argument(
        "--algo",
        default="ppo",
        type=str,
        help="the algorithm to use",
        choices=algo_name_list,
    )

    parser.add_argument("--max_episodes", default=500, type=int)
    parser.add_argument("--episode_length", default=500, type=int)
    parser.add_argument(
        "--map", default=1, type=int, choices=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    )
    parser.add_argument("--shuffle_map", action="store_true")

    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--device", default="cpu", type=str, choices=["cpu", "cuda"])
    parser.add_argument("--save_interval", default=100, type=int)
    parser.add_argument("--render", action="store_true")

    parser.add_argument("--load_model", action="store_true")
    parser.add_argument("--load_run", default=2, type=int)
    parser.add_argument("--load_episode", default=900, type=int)

    return parser.parse_args()
def load_model(algo,run_dir,load_episode,device="cpu"):
    model = algo(device)
    load_dir = os.path.join(run_dir)
    model.load(load_dir, load_episode)
    return model
def choose_agent(episode,dir,p=0.5,device='cpu'):
    # to do : self play
    # p 控制使用的模型是随机的还是
    # if episode<100:
    #     return frozen_agent()
    # if episode<1000:
    #      return random_agent()
    if episode<2000:
        if random.uniform(0,1)<p:
            #return load_model(PPO,dir,episode//100*100,device)
            return load_model(PPO, dir, 500, device)
        else:
            return random_agent()



def main(args):
    pprint(args.__dict__)

    env = get_game(args.seed)
    if not args.shuffle_map:
        env.specify_a_map(
            args.map
        )  # specifying a map, you can also shuffle the map by not doing this step

    num_agents = env.n_player
    print(f"Total agent number: {num_agents}")

    ctrl_agent_index = 1
    print(f"Agent control by the actor: {ctrl_agent_index}")

    width = env.env_core.view_setting["width"] + 2 * env.env_core.view_setting["edge"]
    height = env.env_core.view_setting["height"] + 2 * env.env_core.view_setting["edge"]
    print(f"Game board width: {width}")
    print(f"Game board height: {height}")

    act_dim = env.action_dim
    obs_dim = 25 * 25
    print(f"action dimension: {act_dim}")
    print(f"observation dimension: {obs_dim}")

    setup_seed(args.seed)

    run_dir, log_dir = make_logpath(args.game_name, args.algo)

    print(f"store in {run_dir}") #F:\博弈论\OlympicsProject\rl_trainer\models\olympics-running\ppo\run9
    if not args.load_model:
        writer = SummaryWriter(
            os.path.join(
                str(log_dir),
                "{}_{} on map {}".format(
                    datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
                    args.algo,
                    "all" if args.shuffle_map else args.map,
                ),
            )
        )
        save_config(args, log_dir)
    shutil.copyfile(r'F:\博弈论\OlympicsProject\rl_trainer\main.py', os.path.join(run_dir, 'main.py'))
    record_win = deque(maxlen=100)
    record_win_op = deque(maxlen=100)

    algo = algo_map[args.algo]

    if args.load_model:
        model = algo(args.device)
        load_dir = os.path.join(os.path.dirname(run_dir), "run" + str(args.load_run))
        model.load(load_dir, episode=args.load_episode)
    else:
        model = PPO(args.device, run_dir, writer)  # model is the controlled agent
        Transition = namedtuple(
            "Transition",
            ["state", "action", "a_log_prob", "reward", "next_state", "done"],
        )

    opponent_agent = random_agent()  # we use random opponent agent here

    episode = 0
    train_count = 0
    op_dir = os.path.join(os.path.dirname(run_dir), "run" + str(9))
    with tqdm(range(args.max_episodes)) as pbar:
        while episode < args.max_episodes:
            state = env.reset(args.shuffle_map)

            opponent_agent = choose_agent(episode,op_dir,device=args.device)
            if args.render:
                env.env_core.render()
            obs_ctrl_agent = np.array(state[ctrl_agent_index]["obs"]).flatten()
            obs_oppo_agent = state[1 - ctrl_agent_index]["obs"]
            obs_oppo_agent = np.array(state[1-ctrl_agent_index]["obs"]).flatten()# 为了适应self play的情况
            episode += 1
            pbar.update()
            step = 0
            Gt = 0

            while True:
                action_opponent = opponent_agent.choose_action(
                    obs_oppo_agent
                )  # opponent action
                # action_opponent = [
                #     [0],
                #     [0],
                # ]  # here we assume the opponent is not moving in the demo

                action_ctrl_raw, action_prob = model.select_action(
                    obs_ctrl_agent, False if args.load_model else True
                )
                # inference
                action_ctrl = actions_map[action_ctrl_raw]
                action_ctrl = [[action_ctrl[0]], [action_ctrl[1]]]  # wrapping up the action

                action = (
                    [action_opponent, action_ctrl]
                    if ctrl_agent_index == 1
                    else [action_ctrl, action_opponent]
                )
                next_state, reward, done, _, info = env.step(action)

                next_obs_ctrl_agent = next_state[ctrl_agent_index]["obs"]
                next_obs_oppo_agent = next_state[1 - ctrl_agent_index]["obs"]

                step += 1

                # simple reward shaping
                if not done:
                    post_reward = [-1.0, -1.0]
                else:
                    if reward[0] != reward[1]:
                        post_reward = (
                            [reward[0] - 100, reward[1]]
                            if reward[0] < reward[1]
                            else [reward[0], reward[1] - 100]
                        )
                    else:
                        post_reward = [-1.0, -1.0]

                if not args.load_model:
                    trans = Transition(
                        obs_ctrl_agent,
                        action_ctrl_raw,
                        action_prob,
                        post_reward[ctrl_agent_index],
                        next_obs_ctrl_agent,
                        done,
                    )
                    model.store_transition(trans)

                obs_oppo_agent = next_obs_oppo_agent
                obs_ctrl_agent = np.array(next_obs_ctrl_agent).flatten()
                if args.render:
                    env.env_core.render()
                Gt += reward[ctrl_agent_index] if done else -1

                if done:
                    win_is = (
                        1 if reward[ctrl_agent_index] > reward[1 - ctrl_agent_index] else 0
                    )
                    win_is_op = (
                        1 if reward[ctrl_agent_index] < reward[1 - ctrl_agent_index] else 0
                    )
                    record_win.append(win_is)
                    record_win_op.append(win_is_op)
                    print(
                        "Episode: ",
                        episode,
                        "controlled agent: ",
                        ctrl_agent_index,
                        "; Episode Return: ",
                        Gt,
                        "; win rate(controlled & opponent): ",
                        "%.2f" % (sum(record_win) / len(record_win)),
                        "%.2f" % (sum(record_win_op) / len(record_win_op)),
                        "; Trained episode:",
                        train_count,
                    )

                    if not args.load_model:
                        if args.algo == "ppo" and len(model.buffer) >= model.batch_size:
                            if win_is == 1:
                                model.update(episode)
                                train_count += 1
                            else:
                                model.clear_buffer()

                        writer.add_scalar("training Gt", Gt, episode)

                    break
            if episode % args.save_interval == 0 and not args.load_model:
                model.save(run_dir, episode)


if __name__ == "__main__":
    args = get_args()
    main(args)
