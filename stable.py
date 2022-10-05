
from temp import World


from stable_baselines3 import PPO

env = World()

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)

obs = env.reset()
for i in range(1000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)

    print(reward)
   
    if done:
      obs = env.reset()