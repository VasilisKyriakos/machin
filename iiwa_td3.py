from machin.frame.algorithms import TD3
from machin.utils.logging import default_logger as logger
import torch as t
import torch.nn as nn

from iiwa import Iiwa_World
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pickle
import time

crp, = plt.plot([], [])
rep, = plt.plot([], [])

# configurations
env = Iiwa_World()
observe_dim = 7
action_dim = 7
action_range = 10
max_episodes = 1000
max_steps = 1000
noise_param = (0, 0.2)
noise_mode = "normal"
solved_reward = -100
solved_repeat = 10
sample_freq = 100
demo_count = 10

identifier="iiwa_td3"

tot_reward = []
demos_all=[]


# model definition
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, action_range):
        super().__init__()

        self.fc1 = nn.Linear(state_dim, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, action_dim)
        self.action_range = action_range

    def forward(self, state):
        a = t.relu(self.fc1(state))
        a = t.relu(self.fc2(a))
        a = t.tanh(self.fc3(a)) * self.action_range
        return a


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()

        self.fc1 = nn.Linear(state_dim + action_dim, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, 1)

    def forward(self, state, action):
        state_action = t.cat([state, action], 1)
        q = t.relu(self.fc1(state_action))
        q = t.relu(self.fc2(q))
        q = self.fc3(q)
        return q


if __name__ == "__main__":

    start_time = time.time()

    actor = Actor(observe_dim, action_dim, action_range)
    actor_t = Actor(observe_dim, action_dim, action_range)
    critic = Critic(observe_dim, action_dim)
    critic_t = Critic(observe_dim, action_dim)
    critic2 = Critic(observe_dim, action_dim)
    critic2_t = Critic(observe_dim, action_dim)

    td3 = TD3(
        actor,
        actor_t,
        critic,
        critic_t,
        critic2,
        critic2_t,
        t.optim.Adam,
        nn.MSELoss(reduction="sum"),actor_learning_rate=0.01,
                critic_learning_rate=0.01)

    episode, step, reward_fulfilled = 0, 0, 0
    smoothed_total_reward = 0

    def run_episode():

        total_reward = 0
        terminal = False
        step = 0
        state = t.tensor(env.reset(), dtype=t.float32).view(1, observe_dim)
        tmp_observations = []
        judgements = []
        rewards = []

        while not terminal and step <= max_steps:
            step += 1
            with t.no_grad():
                old_state = state
                # agent model inference
                action = td3.act_with_noise(
                    {"state": old_state}, noise_param=noise_param, mode=noise_mode
                )
                #print(action)
                state, reward, terminal = env.step(action.numpy()[0])
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

        td3.store_episode(tmp_observations)
        tot_reward.append(total_reward)

        if(episode%sample_freq==1):
            #run expected reward
            run_eps = [run_episode()[1] for _ in range(demo_count)]
            demos_all.append(run_eps)


        td3.update()

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
    with open(identifier+str(int(start_time)),"wb") as f:
        pickle.dump(results,f)
        


def heatmap(judgements,rewards):

    update_line(crp,judgements)
    update_line(rep,rewards)
    plt.draw()


def update_line(hl, new_data):
    hl.set_xdata(np.arange(len(new_data)))
    hl.set_ydata(new_data)
    


