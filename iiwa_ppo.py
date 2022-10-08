from machin.frame.algorithms import PPO
from machin.utils.logging import default_logger as logger
from torch.distributions import Categorical
import torch as t
import torch.nn as nn
import gym

import pickle
from iiwa import Iiwa_World
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import time




# configurations
env = Iiwa_World()
observe_dim = 7
action_num = 128
max_episodes = 18
max_steps = 2000
solved_reward = -100
solved_repeat = 1000000
demo_count=3
sample_freq = 4

identifier="iiwappo"

moves= np.vstack([np.eye(7),-np.eye(7)])*10000

def mover(i):
    tt=[int(dig) for dig in "{0:b}".format(i).rjust(7,'0')]
    te=np.array(tt)*2-1
    return te


tot_reward = []
demos_all=[]

# model definition
class Actor(nn.Module):
    def __init__(self, state_dim, action_num):
        super().__init__()

        self.fc1 = nn.Linear(state_dim, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, action_num)

    def forward(self, state, action=None):
        a = t.relu(self.fc1(state))
        a = t.relu(self.fc2(a))
        probs = t.softmax(self.fc3(a),dim=1)
    

        dist = Categorical(probs=probs)
        act = action if action is not None else dist.sample()
        act_entropy = dist.entropy()
        act_log_prob = dist.log_prob(act.flatten())
        return act, act_log_prob, act_entropy


class Critic(nn.Module):
    def __init__(self, state_dim):
        super().__init__()

        self.fc1 = nn.Linear(state_dim, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, 1)

    def forward(self, state):
        v = t.relu(self.fc1(state))
        v = t.relu(self.fc2(v))
        v = self.fc3(v)
        return v


if __name__ == "__main__":
    start_time = time.time()

    
    actor = Actor(observe_dim, action_num)
    critic = Critic(observe_dim)

    ppo = PPO(actor, critic, t.optim.Adam, nn.MSELoss(reduction="sum"))

    episode, step, reward_fulfilled = 0, 0, 0
    smoothed_total_reward = 0

    def run_episode():
            total_reward = 0
            terminal = False
            step = 0
            tmp_observations = []
            state = t.tensor(env.reset(), dtype=t.float32).view(1, observe_dim)
            while not terminal and step <= max_steps:
                step += 1
                with t.no_grad():
                    old_state = state
                    # agent model inference
                    action = ppo.act({"state": old_state})[0]
                    state, reward, terminal = env.step(mover(action.item()))
                    state = t.tensor(state, dtype=t.float32).view(1, observe_dim)
                    total_reward += reward

                    

                    tmp_observations.append(
                        {
                            "state": {"state": old_state},
                            "action": {"action": action},
                            "next_state": {"state": state},
                            "reward": reward,
                            "terminal": terminal or step == max_steps,
                        }
                    )
            return tmp_observations, total_reward

    while episode < max_episodes:

        episode += 1
        tmp_observations, total_reward = run_episode()
        
        # update
        ppo.store_episode(tmp_observations)
        tot_reward.append(total_reward)

        if(episode%sample_freq==1):
            #run expected reward
            run_eps = [run_episode()[1] for _ in range(demo_count)]
            demos_all.append(run_eps)
            

            
        ppo.update()

        # show reward
        smoothed_total_reward = smoothed_total_reward * 0.9 + total_reward * 0.1
        logger.info(f"Episode {episode} total reward={total_reward:.2f}")

        if smoothed_total_reward > solved_reward:
            reward_fulfilled += 1
            if reward_fulfilled >= solved_repeat:
                logger.info("Environment solved!")
                exit(0)
        else:
            reward_fulfilled = 0

    # store results
    duration=(time.time() - start_time)
    results={"dur":duration,"rewards":tot_reward,"evaluations":demos_all}
    with open(identifier,"wb") as f:
        pickle.dump(results,f)
        