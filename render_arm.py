import myosuite
print(myosuite.__file__)

from myosuite.utils import gym

env = gym.make('myoArmReachFixed-v0')

# env.reset()
# for t in range(1000):
#     env.mj_render()
#     _ = env.step(env.action_space.sample())

# breakpoint()

from stable_baselines3 import PPO

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=250_000)

vec_env = model.get_env()
obs = vec_env.reset()
for i in range(1000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    env.mj_render()

breakpoint()