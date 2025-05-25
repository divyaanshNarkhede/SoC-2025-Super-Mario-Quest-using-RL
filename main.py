import gym

env = gym.make("Taxi-v3",render_mode="ansi").unwrapped
env.reset()
state = env.encode(3, 1, 2, 0) # (taxi row, taxi column, passenger index, destination index)


print("State:", state)
env.s = state

print("Action Space {}".format(env.action_space))
print("State Space {}".format(env.observation_space))

print(env.P[328])

output = env.render()
print(output)
