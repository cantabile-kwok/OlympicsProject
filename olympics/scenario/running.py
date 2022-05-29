import itertools

from olympics.core import OlympicsBase
import time
import numpy as np
import json
from math import sqrt


class Running(OlympicsBase):
    def __init__(self, map, seed=None, use_map_dist=False, use_hit_wall=False, use_cross=False):
        super(Running, self).__init__(map, seed)

        self.gamma = 1  # v衰减系数
        self.restitution = 0.5
        self.print_log = False
        self.print_log2 = False
        self.tau = 0.1

        self.speed_cap = 100

        self.draw_obs = True
        self.show_traj = True

        self.use_map_dist = use_map_dist
        self.use_hit_wall = use_hit_wall
        self.use_cross = use_cross

        self.cross_pos = self.store_cross_pos()
        # self.is_render = True

    def check_overlap(self):
        # todo
        pass

    def get_reward(self, cross):

        # print(self.hit_wall[0])  # FIXME

        # ================= overall win/lose reward ==============
        agent_reward = [0.0 for _ in range(self.agent_num)]

        for agent_idx in range(self.agent_num):
            if self.agent_list[agent_idx].finished:
                agent_reward[agent_idx] = 100.0
        # ========================================================

        # ==================== Distance related ==================
        return_dist_rewards = np.array([0 for _ in range(self.agent_num)])
        if self.use_map_dist:
            if hasattr(self, "map_dist"):
                map_dist = self.map_dist
            else:
                # npy_path = f"../olympics/map_dist/map{self.map_num}.npy"
                npy_path = f"olympics/map_dist/map{self.map_num}.npy"

                map_dist = np.load(npy_path)
                print(f"Loaded from {npy_path}")
            for agent_i in range(self.agent_num):
                agent_pos = self.agent_pos[agent_i]
                dist_this_pos = map_dist[int(agent_pos[1]), int(agent_pos[0])]

                if dist_this_pos == 0:
                    all_coordinates = list(itertools.product(range(map_dist.shape[0]), range(map_dist.shape[1])))
                    dist_to_this_pos = [(agent_pos[1] - x) ** 2 + (agent_pos[0] - y) ** 2 for (x, y) in all_coordinates]
                    coordinate_idx_sorted_on_dist = np.argsort(dist_to_this_pos)  # from small to large
                    for coordinate_idx in coordinate_idx_sorted_on_dist:
                        coord = all_coordinates[coordinate_idx]
                        if map_dist[coord] != 0:
                            dist_this_pos = map_dist[coord]
                            break
                    map_dist[int(agent_pos[1]), int(agent_pos[0])] = dist_this_pos

                dist_norm_factor = 100  # NOTE: divide !
                reward_this_pos = -dist_this_pos / dist_norm_factor
                agent_reward[agent_i] += reward_this_pos
                return_dist_rewards[agent_i] = reward_this_pos

            self.map_dist = map_dist
        # ========================================================

        # ================ Hit wall? ===============
        hit_penalty = 8
        if self.use_hit_wall:
            if self.hit_wall[0]:
                self.hit_cnt[0] += 1
                agent_reward[0] -= self.hit_cnt[0] * hit_penalty
            if self.hit_wall[1]:
                self.hit_cnt[1] += 1
                agent_reward[1] -= self.hit_cnt[1] * hit_penalty
            self.hit_wall = [False, False]

        # ==========================================

        # ================= Cross? =================
        if self.use_cross:
            cross_penalty = 10  # NOTE: multiply by this
            if cross:
                cross_list = self.cross_pos['map' + str(self.map_num)]
                cross_pos = [cross_list[i][0] for i in range(len(cross_list))]
                for agent_i in range(self.agent_num):
                    dist = 1e10
                    index = None
                    for i, pos in enumerate(cross_pos):
                        tmp_dist = (pos[0] - self.agent_pos[agent_i][0]) ** 2 + (pos[1] - self.agent_pos[agent_i][1]) ** 2
                        if tmp_dist < dist:
                            dist = tmp_dist
                            index = i
                    crs_pos = cross_list[index]
                    mid = [(crs_pos[1][0] + crs_pos[2][0]) / 2, (crs_pos[1][1] + crs_pos[2][1]) / 2]
                    direction = [crs_pos[0][0] - mid[0], crs_pos[0][1] - mid[1]]
                    weight = direction[0] * self.agent_v[agent_i][0] + direction[1] * self.agent_v[agent_i][1]
                    eps = 1e-7
                    weight /= sqrt(direction[0] ** 2 + direction[1] ** 2) * sqrt(self.agent_v[agent_i][0] ** 2 + self.agent_v[agent_i][1] ** 2) + eps
                    weighted_penalty = weight * cross_penalty
                    agent_reward[agent_i] += weighted_penalty

        # ==========================================

        return agent_reward, return_dist_rewards

    def is_terminal(self):

        if self.step_cnt >= self.max_step:
            return True

        for agent_idx in range(self.agent_num):
            if self.agent_list[agent_idx].finished:
                return True

        return False

    def step(self, actions_list, cross):

        previous_pos = self.agent_pos

        time1 = time.time()
        self.stepPhysics(actions_list, self.step_cnt)
        time2 = time.time()
        # print('stepPhysics time = ', time2 - time1)
        self.speed_limit()

        self.cross_detect(previous_pos, self.agent_pos)

        self.step_cnt += 1
        step_reward, dist_reward = self.get_reward(cross)
        done = self.is_terminal()

        time3 = time.time()
        obs_next = self.get_obs()
        time4 = time.time()
        # print('render time = ', time4-time3)
        # obs_next = 1
        # self.check_overlap()
        self.change_inner_state()

        return obs_next, step_reward, done, "", dist_reward

    def store_cross_pos(self):
        cross_pos = dict()
        with open('olympics/maps.json') as f:
            map_dict = json.load(f)
        for k in map_dict.keys():
            cross_pos[k] = []
            cross_obj = map_dict[k]['cross']['objects']
            keys = list(cross_obj.keys())
            if len(keys) > 1:
                for num in range(1, len(keys), 2):
                    key_1 = keys[num]
                    key_2 = keys[num + 1]
                    cross_pos[k].append([cross_obj[key_1]['initial_position'][1], cross_obj[key_1]['initial_position'][0], cross_obj[key_2]['initial_position'][0]])
        return cross_pos