#Following this tutorial: https://youtu.be/ZC0gMhYhwW0?si=yCyBDWAcmAsPlP-i&t=303
#Cart Pole documentaiton: https://gymnasium.farama.org/environments/classic_control/cart_pole/ 

import gymnasium
env = gymnasium.make("CartPole-v1", render_mode="human")

observation = env.reset() #In cart pole, observation is a vector, will be input to NN.
#The output of the NN will be whatever action we want to take.
#Observation is a vector of 4 values:
# [cart position, cart velocity, pole angle, pole velocity at tip]

print(observation)
print("****")
print(env.action_space)

done = False
while not done:
    observation, reward, terminated, truncated, info = env.step(env.action_space.sample())
    print(env.action_space.sample())

    env.render()

env.close()