import gymnasium as gym

env = gym.make("LunarLander-v3", render_mode="human")

obs, info = env.reset()

terminated = False
truncated = False

while not (terminated or truncated):
    action = env.action_space.sample()

    obs, reward, terminated, truncated, info = env.step(action)

env.close()