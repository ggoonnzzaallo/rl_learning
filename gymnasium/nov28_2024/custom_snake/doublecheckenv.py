from snakeenv import SnekEnv

env = SnekEnv()
episodes = 50   

for episode in range(episodes):
    done = False
    obs = env.reset()
    while not done:
        random_action = env.action_space.sample()
        print(f"Action: {random_action}")
        obs, reward, done, _, info = env.step(random_action)
        print(f"Episode: {episode}, Reward: {reward}")
