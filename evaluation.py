import numpy as np
import random

from tqdm import tqdm
from env.chooseenv import make
from tabulate import tabulate
import argparse
from rl_trainer.algo import *

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


def get_join_actions(state, agent_list):
    joint_actions = []

    for agent_idx in range(len(agent_list)):
        obs = np.array(state[agent_idx])
        actions_raw = agent_list[agent_idx].choose_action(obs)
        if np.isscalar(actions_raw):
            actions = actions_map[actions_raw]
            joint_actions.append([[actions[0]], [actions[1]]])
        else:
            joint_actions.append(actions_raw)
    return joint_actions


def run_game(env, algo_list, agent_list, episode, shuffle_map, map_num, render):
    total_reward = np.zeros(2)
    num_win = np.zeros(3)  # agent 1 win, agent 2 win, draw
    episode = int(episode)
    for i in tqdm(range(1, int(episode) + 1)):
        episode_reward = np.zeros(2)

        state = env.reset(shuffle_map)
        state_buffers = [[np.zeros((25, 25)) for _ in range(args.num_frame - 1)] for i in range(len(agent_list))]
        for j in range(len(agent_list)):
            obs_agent = np.array(state[j]["obs"])
            state_buffers[j].insert(0, obs_agent)

        if render:
            env.env_core.render()

        step = 0

        while True:
            # print(step, end='\t')
            joint_action = get_join_actions(state_buffers, agent_list)
            next_state, reward, done, _, info = env.step(joint_action)
            reward = np.array(reward)
            episode_reward += reward
            if render:
                env.env_core.render()

            if done:
                if reward[0] != reward[1]:
                    if reward[0] == 100:
                        num_win[0] += 1
                    elif reward[1] == 100:
                        num_win[1] += 1
                    else:
                        print('both have not reached 100 reward')
                        # raise NotImplementedError
                        # FIXME
                else:
                    num_win[2] += 1
                break

            for k in range(len(agent_list)):
                next_obs_agent = next_state[k]["obs"]
                obs = np.array(next_obs_agent)
                state_buffers[k].pop(-1)
                state_buffers[k].insert(0, obs)

            step += 1
        total_reward += episode_reward
    total_reward /= episode
    print("total reward: ", total_reward)
    print("Result in map {} within {} episode:".format(map_num, episode))

    header = ["Name", algo_list[0], algo_list[1]]
    data = [
        ["score", np.round(total_reward[0], 2), np.round(total_reward[1], 2)],
        ["win", num_win[0], num_win[1]],
    ]
    print(tabulate(data, headers=header, tablefmt="pretty"))


algo_name_list = ["ppo", "random"]
algo_list = [PPO, random_agent]
algo_map = dict(zip(algo_name_list, algo_list))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--my_ai",
        default="ppo",
        help="[your algo name]/random",
        choices=["ppo", "random"],
    )
    parser.add_argument("--my_ai_run_dir", default="")
    parser.add_argument("--my_ai_run_episode", default=0)
    parser.add_argument("--opponent", default="random", help="[your algo name]/random")
    parser.add_argument("--opponent_run_dir", default="")
    parser.add_argument("--opponent_run_episode", default=0)
    parser.add_argument("--episode", default=20)
    parser.add_argument(
        "--map",
        default="all",
    )
    parser.add_argument("--render", type=bool, default=True)
    parser.add_argument("--seed", default=123)

    parser.add_argument('--actor_hidden_layers', type=int, default=2)
    parser.add_argument('--critic_hidden_layers', type=int, default=2)
    parser.add_argument("--num_frame", default=3, type=int, help="number of frames(states) in one time step")
    parser.add_argument("--use_cnn", action='store_true', help="whether use cnn network")
    args = parser.parse_args()

    env_type = "olympics-running"
    game = make(env_type, conf=None, seed=args.seed)

    if args.map != "all":
        game.specify_a_map(int(args.map))
        shuffle = False
    else:
        shuffle = True

    algo_list = [args.my_ai, args.opponent]  # your are controlling agent purple

    agent_list = []

    if args.my_ai != "random":
        algo = algo_map[args.my_ai]
        algo.use_cnn = args.use_cnn
        if algo.use_cnn:
            algo.num_frame = args.num_frame
        else:
            algo.state_space = args.num_frame * 625
        agent = algo(actor_hidden_layers=args.actor_hidden_layers,
                                     critic_hidden_layers=args.critic_hidden_layers)
        agent.load(args.my_ai_run_dir, int(args.my_ai_run_episode))
        agent_list.append(agent)
    else:
        agent_list.append(random_agent(args.seed))

    if args.opponent != "random":
        agent = algo_map[args.opponent](actor_hidden_layers=args.actor_hidden_layers,
                                        critic_hidden_layers=args.critic_hidden_layers)
        agent.load(args.opponent_run_dir, int(args.opponent_run_episode))
        agent_list.append(agent)
    else:
        agent_list.append(random_agent(args.seed))

    # NOTE: [our ai, random]

    run_game(
        game,
        algo_list=algo_list,
        agent_list=agent_list,
        episode=args.episode,
        shuffle_map=shuffle,
        map_num=args.map,
        render=args.render,
    )
