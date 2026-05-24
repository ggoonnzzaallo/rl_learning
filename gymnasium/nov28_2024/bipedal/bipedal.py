#Following this tutorial: https://youtu.be/ZC0gMhYhwW0?si=yCyBDWAcmAsPlP-i&t=303
#Cart Pole documentaiton: https://gymnasium.farama.org/environments/box2d/bipedal_walker/

import gymnasium
env = gymnasium.make("BipedalWalker-v3", render_mode="human")

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


# The observation space looks like this:
'''
(array([ 2.7452081e-03,  1.4362499e-05, -1.8727871e-03, -1.6000094e-02,
        9.2613742e-02,  4.3475609e-03,  8.5965359e-01, -2.0542878e-03,
        1.0000000e+00,  3.2896299e-02,  4.3473891e-03,  8.5347074e-01,
       -2.9581243e-03,  1.0000000e+00,  4.4081330e-01,  4.4581941e-01,
        4.6142203e-01,  4.8954940e-01,  5.3410190e-01,  6.0246003e-01,
        7.0914775e-01,  8.8593036e-01,  1.0000000e+00,  1.0000000e+00],
      dtype=float32), {})
'''

# The env.action_space looks like this:
# Box(-1.0, 1.0, (4,), float32)


# Observation space explanation (24 values total):
#
# Hull (Body) Information [0-3]:
#   [0] Hull angle
#   [1] Hull angular velocity
#   [2] Hull horizontal speed
#   [3] Hull vertical speed
#
# Legs Information [4-13]:
#   First Leg [4-8]:
#     [4] Ground contact flag (1.0 if touching, 0.0 if not)
#     [5] Joint 1 angle
#     [6] Joint 1 speed
#     [7] Joint 2 angle
#     [8] Joint 2 speed
#   Second Leg [9-13]:
#     [9] Ground contact flag (1.0 if touching, 0.0 if not)
#     [10] Joint 1 angle
#     [11] Joint 1 speed
#     [12] Joint 2 angle
#     [13] Joint 2 speed
#
# Lidar Readings [14-23]:
#   [14-23] 10 lidar rangefinder measurements
#   Values normalized to [0, 1]
#
# Action space (4 values, all between -1 and 1):
#   [0] First leg hip joint motor torque
#   [1] First leg knee joint motor torque
#   [2] Second leg hip joint motor torque
#   [3] Second leg knee joint motor torque