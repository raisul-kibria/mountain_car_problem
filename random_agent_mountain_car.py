#!/usr/bin/env python

# MIT License
# Copyright (c) 2017 Massimiliano Patacchiola
# https://mpatacchiola.github.io/blog/
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from utils.mountain_car import MountainCar
import random
import numpy as np

my_car = MountainCar(mass=1000.0, friction=0.3, delta_t=0.1)
cumulated_reward = 0
print("Starting random agent...")
for step in range(100):
    action = random.randint(a=0, b=2)
    observation, reward, done = my_car.step(action)
    cumulated_reward += reward
    if done: break

print("Finished after: " + str(step+1) + " steps")
print("Cumulated Reward: " + str(cumulated_reward))
print("Saving the gif in: ./outputs/random_agent_general/mountain_car_random.gif")
my_car.render(file_path='./outputs/random_agent_general/mountain_car_random.gif', mode='gif')
print("Complete!")
# my_car = MountainCar(mass=1000.0, friction=0.3, delta_t=0.1)
# cumulated_reward = 0
# p = np.load('policy.npy')
# tot_bins = 11
# velocity_state_array = np.linspace(-1.5, +1.5, num=tot_bins-1, endpoint=False)
# position_state_array = np.linspace(-1.2, +0.5, num=tot_bins-1, endpoint=False)
# print("Starting random agent...")
# for step in range(100):
#     if step == 0:
#         action = random.randint(a=0, b=2)
#         observation = my_car.reset(exploring_starts=True)
#         reward = 0
#         done = False
#     else:
#         observation, reward, done = my_car.step(action)
#     observation = (np.digitize(observation[1], velocity_state_array), 
#                        np.digitize(observation[0], position_state_array))
#     action = int(p[observation[0], observation[1]])
#     cumulated_reward += reward
#     if done: break