import gymnasium as gym
from Gymnasium_Snake_Game import gymnasium_snake_game

print(1)
env = gym.make('Snake-0', render_mode = 'human')
env.reset