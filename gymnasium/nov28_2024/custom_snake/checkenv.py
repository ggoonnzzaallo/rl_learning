from stable_baselines3.common.env_checker import check_env
from snakeenv import SnekEnv

env = SnekEnv()
check_env(env) #It will check your custom environment and output additional warnings if needed.