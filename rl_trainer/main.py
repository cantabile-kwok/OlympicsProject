# import _io
import argparse
import datetime
import os.path
import random
import sys
from copy import deepcopy
from pathlib import Path
# from pprint import pprint
from typing import Dict
from tqdm import tqdm
import shutil
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

base_dir = str(Path(__file__).resolve().parent.parent)
sys.path.append(base_dir)
sys.path.append(os.path.abspath('./'))
if "/mnt/lustre/sjtu/home/ywg12/remote/code/BottomUpAttention/bottom-up-attention.pytorch-master" in sys.path:
    sys.path.pop(sys.path.index("/mnt/lustre/sjtu/home/ywg12/remote/code/BottomUpAttention/bottom-up-attention.pytorch-master"))


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

# <<<<<<< HEAD
BEGIN_SAVE = 300



def get_game(seed: int = None, config: Dict = None, log_file=None):
    return make("olympics-running", seed, config, log_file=log_file)


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

    parser.add_argument("--max_episodes", default=3000, type=int)
    parser.add_argument("--episode_length", default=1000, type=int)
    parser.add_argument(
        "--map", default=1, type=int, choices=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    )
    parser.add_argument("--shuffle_map", action="store_true")
    parser.add_argument("--buffer_capacity", default=3000, type=int)

    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--device", default="cpu", type=str, choices=["cpu", "cuda"])
    parser.add_argument("--save_interval", default=100, type=int)
    parser.add_argument("--render", action="store_true")

    parser.add_argument("--load_model", action="store_true")
    parser.add_argument("--load_run", default=2, type=int)
    parser.add_argument("--load_episode", default=900, type=int)
    parser.add_argument("--run_dir", type=str, help='Running directory name (for experiments)')
    parser.add_argument('--actor_hidden_layers', type=int, default=2)
    parser.add_argument('--critic_hidden_layers', type=int, default=2)
    parser.add_argument("--num_frame", default=1, type=int, help="number of frames(states) in one time step")
    parser.add_argument("--use_cnn", action='store_true', help="whether use cnn network")
    parser.add_argument('--train_by_win', action='store_true')
    parser.add_argument("--use_step_dist", action='store_true')
    parser.add_argument('--shuffle_place', action='store_true', help="whether shuffle start place")

    return parser.parse_args()


# <<<<<<< HEAD
# =======
def load_model(algo, run_dir, load_episode, device="cpu"):
    model = algo(device)
    load_dir = os.path.join(run_dir)
    model.load(load_dir, load_episode)
    return model


def choose_agent(episode, onlinemodel, pool:agent_pool, p=0.5, device='cpu', args=None):
    # TODO: add args into sample()
    # to do : self play
    # online model 当前训练的模型
    # pool 历史模型池
    # p 控制使用的模型是随机的还是
    # if episode<100:
    #     return frozen_agent()
    if episode < 100:
        return frozen_agent(), -1
    if episode < 500:
        return random_agent(), -1
    if episode < 2000:
        if random.uniform(0, 1) < p:
            # return load_model(PPO,dir,episode//100*100,device)
            return pool.sample(args=args)
        else:
            return onlinemodel, -1


# >>>>>>> main
def main(args):
    run_dir, log_dir = make_logpath(args.game_name, args.algo)
    run_dir = os.path.join(os.path.dirname(run_dir), args.run_dir)
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)
    log_dir = run_dir

    log_file = open(f"{log_dir}/train.log", 'w')

    print(args.__dict__, file=log_file)

    env = get_game(args.seed, log_file=log_file)
    if not args.shuffle_map:
        env.specify_a_map(
            args.map
        )  # specifying a map, you can also shuffle the map by not doing this step

    num_agents = env.n_player
    print(f"Total agent number: {num_agents}", file=log_file)

    ctrl_agent_index = 1
    print(f"Agent control by the actor: {ctrl_agent_index}", file=log_file)

    width = env.env_core.view_setting["width"] + 2 * env.env_core.view_setting["edge"]
    height = env.env_core.view_setting["height"] + 2 * env.env_core.view_setting["edge"]
    print(f"Game board width: {width}", file=log_file)
    print(f"Game board height: {height}", file=log_file)

    act_dim = env.action_dim
    obs_dim = 25 * 25
    print(f"action dimension: {act_dim}", file=log_file)
    print(f"observation dimension: {obs_dim}", file=log_file)

    setup_seed(args.seed)

    print(f"store in {run_dir}", file=log_file)
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

    shutil.copyfile('rl_trainer/main.py', os.path.join(run_dir, 'main.py'))
    record_win = deque(maxlen=100)
    record_win_op = deque(maxlen=100)

    algo = algo_map[args.algo]
    algo.use_cnn = args.use_cnn
    print(f"Use CNN: {args.use_cnn}", file=log_file)
    if algo.use_cnn:
        algo.num_frame = args.num_frame
    else:
        algo.state_space = args.num_frame * 625

    if args.load_model:
        model = algo(args.device)
        load_dir = os.path.join(os.path.dirname(run_dir), "run" + str(args.load_run))
        model.load(load_dir, episode=args.load_episode)
    else:
        model = PPO(args.device, run_dir, writer,
                    actor_hidden_layers=args.actor_hidden_layers,
                    critic_hidden_layers=args.critic_hidden_layers)  # model is the controlled agent
        Transition = namedtuple(
            "Transition",
            ["state", "action", "a_log_prob", "reward", "next_state", "done"],
        )

    opponent_agent = random_agent()  # we use random opponent agent here
    Agent_pool = agent_pool(args.device)

    episode = 0
    train_count = 0
# <<<<<<< HEAD
# =======

    # ================ NOTE: optionally add an existing model into agent pool =================
    # op_dir = os.path.join(os.path.dirname(run_dir), "run" + str(9))  # use run9,just for test
    # Agent_pool.add(op_dir, 500)
    # =========================================================================================
# >>>>>>> main

    with tqdm(range(args.max_episodes)) as pbar:
        while episode < args.max_episodes:
            state = env.reset(args.shuffle_map)
            state_buffer = [np.zeros((25, 25)) for _ in range(args.num_frame - 1)]

            state_buffer_for_oppo = [np.zeros((25, 25)) for _ in range(args.num_frame - 1)]

            if args.shuffle_place:
                ctrl_agent_index = random.randint(0, 1)

            opponent_agent, index = choose_agent(episode, onlinemodel=model,
                                                 pool=Agent_pool,
                                                 device=args.device,
                                                 args=args
                                                 )
            # when index =-1 ，说明未从pool中取
            if args.render:
                env.env_core.render()
            obs_ctrl_agent = np.array(state[ctrl_agent_index]["obs"])
            state_buffer.insert(0, obs_ctrl_agent)

            obs_oppo_agent = np.array(state[1 - ctrl_agent_index]["obs"])  # 为了适应self play的情况
            state_buffer_for_oppo.insert(0, obs_oppo_agent)

            episode += 1
            pbar.update()
            step = 0
            Gt = 0

            while True:
                action_opponent = opponent_agent.choose_action(
                    np.array(state_buffer_for_oppo)
                )  # opponent action'
                if isinstance(action_opponent, int):  # for ppo opponent
                    action_opponent = actions_map[action_opponent]
                    action_opponent = [[action_opponent[0]], [action_opponent[1]]]
                # action_opponent = [
                #     [0],
                #     [0],
                # ]  # here we assume the opponent is not moving in the demo

                action_ctrl_raw, action_prob = model.select_action(
                    np.array(state_buffer), False if args.load_model else True
                )
                # inference
                action_ctrl = actions_map[action_ctrl_raw]
                action_ctrl = [[action_ctrl[0]], [action_ctrl[1]]]  # wrapping up the action

                action = (
                    [action_opponent, action_ctrl]
                    if ctrl_agent_index == 1
                    else [action_ctrl, action_opponent]
                )
                next_state, reward, done, _, info, dist_reward = env.step(action, np.array(state_buffer).__contains__(4))

                next_obs_ctrl_agent = next_state[ctrl_agent_index]["obs"]
                next_obs_oppo_agent = next_state[1 - ctrl_agent_index]["obs"]

                step += 1

                # simple reward shaping
                if not done:
                    if not args.use_step_dist:
                        post_reward = [-1.0, -1.0]  # NOTE: non relevant to dist
                    else:
                        post_reward = reward  # NOTE: adopt step-level dist reward

                else:
                    if reward[0] != reward[1]:
                        post_reward = (
                            [reward[0] - 100, reward[1]]
                            if reward[0] < reward[1]
                            else [reward[0], reward[1] - 100]
                        )
                    else:
                        post_reward = [-1.0, -1.0]

                obs_oppo_agent = np.array(next_obs_oppo_agent)
                obs_ctrl_agent = np.array(next_obs_ctrl_agent)
                last_state = deepcopy(state_buffer)
                state_buffer.pop(-1)
                state_buffer.insert(0, obs_ctrl_agent)
                state_buffer_for_oppo.pop(-1)
                state_buffer_for_oppo.insert(0, obs_oppo_agent)


                if not args.load_model:
                    trans = Transition(
                        np.array(last_state),
                        action_ctrl_raw,
                        action_prob,
                        post_reward[ctrl_agent_index],
                        np.array(state_buffer),
                        done,
                    )
                    model.store_transition(trans)

                if args.render:
                    env.env_core.render()
                Gt += reward[ctrl_agent_index] if done else -1

                if done:
                    win_is = (
                        1 if dist_reward[ctrl_agent_index] > dist_reward[1 - ctrl_agent_index] else 0
                    )
                    win_is_op = (
                        1 if dist_reward[ctrl_agent_index] < dist_reward[1 - ctrl_agent_index] else 0
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
                        ";Result:",
                        (win_is, win_is_op),
                        file=log_file,
                        flush=True
                    )
                    # win_r = sum(record_win) / len(record_win)
                    win_r = int(win_is > win_is_op)
                    print(win_r, file=log_file)
                    # win_r = 0.6 #just for test
                    if win_r > 0.5 and index >= 0:
                        # update pool
                        Agent_pool.update(index, win_r)
# >>>>>>> main
                    if not args.load_model:
                        if args.algo == "ppo":
                            if args.train_by_win:
                                if win_is == 1:
                                    model.merge_buffer()
                                    if model.counter > args.buffer_capacity:
                                        model.update(episode)
                                        train_count += 1
                                    model.clear_tmp_buffer()
                                else:
                                    model.clear_tmp_buffer()
                            else:
                                model.merge_buffer()
                                if model.counter >= args.buffer_capacity:
                                    model.update(episode)
                                    train_count += 1
                                model.clear_tmp_buffer()

                        writer.add_scalar("training Gt", Gt, episode)

                    break
            if episode % args.save_interval == 0 and not args.load_model:
                model.save(run_dir, episode)
                if episode >= BEGIN_SAVE:
                    Agent_pool.add(run_dir, episode)
                log_file.flush()

    log_file.close()


if __name__ == "__main__":
    args = get_args()
    main(args)
