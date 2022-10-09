from functools import partial
from inspect import unwrap
from operator import concat
from machin.frame.algorithms import PPO
from machin.utils.logging import default_logger as logger
from torch.distributions import Categorical
import torch as t
import torch.nn as nn

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from pendulum import World
import time
import pickle


# configurations
identifier="pendulum_ppo"
env = World()
tot_reward = []
demos_all=[]
ppo = None


observe_dim = 3
action_num = 2
max_episodes = 5000
max_steps = 4000
solved_reward = 50000
solved_repeat = 1000000
speed = 10

demo_count = 10
sample_freq = 100


def heatmap(harvest,harvest2,im1,im2):

    im1.set_data(harvest)
    im2.set_data(harvest2)
    im1.set_clim(np.min(harvest),np.max(harvest))
    im2.set_clim()
    plt.draw()



def debugnn(network):
    arr = np.zeros([50,50])
    for i in range(50):
        for j in range(50):

            si = (i/50)*2*np.pi
            sj = (j/24.5)-1
            state = [np.cos(si),np.sin(si),sj]
            state = t.tensor(state, dtype=t.float32).view(1,observe_dim)
            res = network.forward(state)
            arr[j,i] = res.detach()

    return arr
      


def debugActor(network):
    arr = np.zeros([50,50])
    for i in range(50):
        for j in range(50):

            si = (i/50)*2*np.pi
            sj = (j/24.5)-1
            state = [np.cos(si),np.sin(si),sj]
            state = t.tensor(state, dtype=t.float32).view(1,observe_dim)
            res = network.forward(state)[0]
            arr[j,i] = res.detach()
    
    return arr


def debugppo(critic,actor,im1,im2):
        heatmap(debugnn(critic),debugActor(actor),im1,im2)





# model definition
class Actor(nn.Module):

    def __init__(self, state_dim, action_num):

        super().__init__()
        self.fc1 = nn.Linear(state_dim, 5)
        self.fc2 = nn.Linear(5, 5)
        self.fc3 = nn.Linear(5, action_num)


    def forward(self, state, action=None):

        a = (self.fc1(state))
        a = (self.fc2(a))
        probs = t.softmax(self.fc3(a), dim=1)
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




""" def initials():

    actor = Actor(observe_dim, action_num)
    critic = Critic(observe_dim)
    fig, (ax1,ax2) = plt.subplots(1,2)

    im1 = ax1.imshow(np.ones([50,50]),extent=[-1,1,1,-1],cmap='RdBu_r')
    im2 = ax2.imshow(np.zeros([50,50]),extent=[-1,1,1,-1],cmap = 'viridis')
    ax1.set_title('Critic')
    ax2.set_title('Actor')
    fig.colorbar(im1,ax=ax1)
    fig.colorbar(im2,ax=ax2)
    plt.ylabel('velocity')
    plt.xlabel('position')
    plt.show(block=False)

    ppo = PPO(actor, critic, t.optim.Adam, nn.MSELoss(reduction="sum"),actor_learning_rate=0.01,

                critic_learning_rate=0.01)

    return ppo, im1, im2, actor, critic """




if __name__ == "__main__":

    start_time = time.time()


    actor = Actor(observe_dim, action_num)
    critic = Critic(observe_dim)
    fig, (ax1,ax2) = plt.subplots(1,2)

    im1 = ax1.imshow(np.ones([50,50]),extent=[-1,1,1,-1],cmap='RdBu_r')
    im2 = ax2.imshow(np.zeros([50,50]),extent=[-1,1,1,-1],cmap = 'viridis')
    ax1.set_title('Critic')
    ax2.set_title('Actor')
    fig.colorbar(im1,ax=ax1)
    fig.colorbar(im2,ax=ax2)
    plt.ylabel('velocity')
    plt.xlabel('position')
    plt.show(block=False)

    ppo = PPO(actor, critic, t.optim.Adam, nn.MSELoss(reduction="sum"),actor_learning_rate=0.01,

                critic_learning_rate=0.01)

    episode, step, reward_fulfilled = 0, 0, 0
    smoothed_total_reward = 0
    episode=0


    def run_episode():

        step = 0
        total_reward = 0
        terminal = False
        state = t.tensor(env.reset(), dtype=t.float32).view(1, observe_dim)
        tmp_observations = []

        while not terminal and step <= max_steps:

            step += 1

            with t.no_grad():

                old_state = state
                # agent model inference
                action = ppo.act({"state": old_state})
                action = action[0] #comment
                robaction = [-speed,speed][action.detach()[0]]
                state, reward, terminal = env.step([robaction])
                reward=reward[0]
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

        return total_reward , tmp_observations


    while episode < max_episodes:

        episode += 1
        total_reward,tmp_observations = run_episode()

        # update
        ppo.store_episode(tmp_observations)
        tot_reward.append(total_reward)

        if episode % 1 == 0:
            debugppo(critic,actor,im1,im2)
            plt.pause(0.01)

        if(episode%sample_freq==1):
            #run expected reward
            run_eps = [run_episode()[1] for _ in range(demo_count)]
            demos_all.append(run_eps)


        ppo.update()

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
    with open(identifier+str(int(start_time)),"wb") as f:
        pickle.dump(results,f)



