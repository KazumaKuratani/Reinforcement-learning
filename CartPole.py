# %%
import gym
env = gym.make("CartPole-v0")
# %%
state = env.reset()
print(state)

action_space = env.action_space
print(action_space)
# %%
