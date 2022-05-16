import sys
from os import path

father_path = path.dirname(__file__)
sys.path.append(str(father_path))
from generator import create_scenario
import argparse
from agent import *
import time
from scenario.running import Running
from rl_trainer.algo.ppo import PPO


import random
import numpy as np
import matplotlib.pyplot as plt
import json


def store(record, name):
    with open("logs/" + name + ".json", "w") as f:
        f.write(json.dumps(record))


def load_record(path):
    file = open(path, "rb")
    filejson = json.load(file)
    return filejson


RENDER = True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--map", default="map4", type=str, help="map1/map2/map3/map4")
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--my_ai_run_dir", default="")
    parser.add_argument("--my_ai_run_episode", default=0)
    parser.add_argument(
        "--my_ai",
        default="ppo",
        help="[your algo name]/random",
        choices=["ppo", "random"],
    )

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    # agent1 = random_agent()
    agent2 = random_agent()

    algo_name_list = ["ppo", "random"]
    algo_list = [PPO, random_agent]
    algo_map = dict(zip(algo_name_list, algo_list))

    agent1 = algo_map[args.my_ai]()
    agent1.load(args.my_ai_run_dir, int(args.my_ai_run_episode))
    # agent3 = random_agent()


    map_index_seq = list(range(1, 5))
    time_s = time.time()
    for i in range(20):
        print("==========================================")
        ind = map_index_seq.pop(0)
        print("map index: ", ind)
        Gamemap = create_scenario("map" + str(ind))
        map_index_seq.append(ind)

        rnd_seed = random.randint(0, 1000)
        game = Running(Gamemap, seed=rnd_seed)
        game.map_num = ind

        obs = game.reset()
        if RENDER:
            game.render()

        done = False
        step = 0
        if RENDER:
            game.render("MAP {}".format(ind))

        while not done:
            step += 1

            action1 = agent1.choose_action(obs[0].flatten(), train=False)
            from evaluation import actions_map
            action1 = actions_map[action1]
            action2 = agent2.act(obs[1])
            # action2 = agent2.act(obs[1])

            obs, reward, done, _ = game.step([action1, action2])
            print("Action:", action1)
            print('-' * 89)

            if RENDER:
                game.render()

        print("Episode Reward = {}".format(reward))
