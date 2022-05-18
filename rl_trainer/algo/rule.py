# TODO: you can implement a rule-based agent to compete with.

import random

class frozen_agent:
    def __init__(self, seed=None):
        self.force_range = [-100, 200]
        self.angle_range = [-30, 30]

    def select_action(self, obs, _=False):
        force = random.uniform(self.force_range[0], self.force_range[1])
        angle = random.uniform(self.angle_range[0], self.angle_range[1])

        return [[0], [0]]
    def choose_action(self, state, train=False):
        return [[0], [0]]