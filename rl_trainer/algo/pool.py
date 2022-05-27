import numpy as np
import os
from rl_trainer.algo.ppo import PPO
import torch
def load_model(run_dir,load_episode,device="cpu",algo=PPO):
    model = algo(device)
    load_dir = os.path.join(run_dir)
    model.load(load_dir, load_episode)
    return model
class agent_pool(object):
    def __init__(self, device):
        self.device = device
        self.pool = {}
        self.quality_score = []
        self.prob = []
        self.index = 0
        self.alpha = 0.01
    def add(self,dir,episode):
        self.pool[self.index]={'dir':dir,'episode':episode}
        self.quality_score.append(1.0)
        self.prob = torch.softmax(torch.tensor(self.quality_score),0).tolist()
        self.index += 1



    def sample(self):
        index = np.random.choice(self.index,p=np.array(self.prob)/np.array(self.prob).sum())
        dir = self.pool[index]['dir']
        episode = self.pool[index]['episode']
        return load_model(dir,episode,device=self.device),index
    def reset(self):
        self.pool = {}
        self.quality_score = []
        self.prob = []
    def update(self,index,win_r):
        self.quality_score[index] -= self.alpha/((self.index+1)*self.prob[index])*(win_r - 0.5)
        print(self.prob)
        self.prob = torch.softmax(torch.tensor(self.quality_score), 0).tolist()
        print(self.prob)
        pass

