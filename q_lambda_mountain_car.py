#!/usr/bin/env python

#MIT License
#Copyright (c) 2017 Massimiliano Patacchiola
#
#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:
#
#The above copyright notice and this permission notice shall be included in all
#copies or substantial portions of the Software.
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE.


import os
import numpy as np
from utils.mountain_car import MountainCar, PARAMS
from utils.utils import calculate_longest_streak, plot_curve, print_policy, create_output_dir
import matplotlib.pyplot as plt
import argparse
import tqdm

def update_state_action(state_action_matrix, trace_matrix, visit_counter_matrix, observation, new_observation,
                   action, reward, alpha, gamma, lambda_, tot_bins):
    '''Return the updated utility matrix

    @param state_action_matrix the matrix before the update
    @param observation the state obsrved at t
    @param new_observation the state observed at t+1
    @param action the action at t
    @param new_action the action at t+1
    @param reward the reward observed after the action
    @param alpha the ste size (learning rate)
    @param gamma the discount factor
    @return the updated state action matrix
    '''
    #Getting the values of Q at t and at t+1
    col = observation[1] + (observation[0]*tot_bins)
    q = state_action_matrix[action, col]
    col_t1 = new_observation[1] + (new_observation[0]*tot_bins)
    q_t1 = np.max(state_action_matrix[:, col_t1])
    #Calculate alpha based on how many time it
    #has been visited
    alpha_counted = 1.0 / (1.0 + visit_counter_matrix[action, col])
    #Applying the update rule
    delta = (reward + gamma * q_t1 - q)

    # replacing and clearing traces
    trace_matrix[:, col] = 0
    trace_matrix[int(action), col] = 1

    state_action_matrix += alpha * delta * trace_matrix
    trace_matrix = update_eligibility(trace_matrix, gamma, lambda_)
    return state_action_matrix

def update_eligibility(trace_matrix, gamma, lambda_):
    """"update the elgibility trace_matrix for the current timestep.
    @param trace_matrix trace_matrix with same dimesions as state_action matrix
    @param gamma eligibility trace decay parameter
    @param lambda_ discount factor for immediate and future rewards.

    @return updated trace_matrix
    """
    trace_matrix *= (gamma * lambda_) # this is a matrix operation!!
    return trace_matrix

def update_visit_counter(visit_counter_matrix, observation, action, tot_bins):
    '''Update the visit counter

    Counting how many times a state-action pair has been
    visited. This information can be used during the update.
    @param visit_counter_matrix a matrix initialised with zeros
    @param observation the state observed
    @param action the action taken
    '''
    col = observation[1] + (observation[0]*tot_bins)
    visit_counter_matrix[action ,col] += 1.0
    return visit_counter_matrix

def update_policy(policy_matrix, state_action_matrix, observation, tot_bins):
    '''Return the updated policy matrix

    @param policy_matrix the matrix before the update
    @param state_action_matrix the state-action matrix
    @param observation the state obsrved at t
    @return the updated state action matrix
    '''
    col = observation[1] + (observation[0]*tot_bins)
    #Getting the index of the action with the highest utility
    best_action = np.argmax(state_action_matrix[:, col])
    #Updating the policy
    policy_matrix[observation[0], observation[1]] = best_action
    return policy_matrix

def return_epsilon_greedy_action(policy_matrix, observation, epsilon=0.1):
    '''Return an action choosing it with epsilon-greedy

    @param policy_matrix the matrix before the update
    @param observation the state obsrved at t
    @param epsilon the value used for computing the probabilities
    @return the updated policy_matrix
    '''
    tot_actions = int(np.nanmax(policy_matrix) + 1)
    action = int(policy_matrix[observation[0], observation[1]])
    non_greedy_prob = epsilon / tot_actions
    greedy_prob = 1 - epsilon + non_greedy_prob
    weight_array = np.full((tot_actions), non_greedy_prob)
    weight_array[action] = greedy_prob
    return int(np.random.choice(tot_actions, 1, p=weight_array))

def return_decayed_value(starting_value, minimum_value, global_step, decay_step, epsilon_strategy):
    """Returns the decayed value.

    decay: 
        decayed_value = starting_value * decay_rate ^ (global_step / decay_steps)
    linear decay: 
        decayed_value = starting_value - ((starting_value - minimum_value) / (decay_steps - 1)) * (global_step)
    @param starting_value the value before decaying
    @param minimum_value acceptable as epsilon
    @param global_step the global step to use for decay (positive integer)
    @param decay_step the step at which the value is decayed
    @param epsilon_strategy: [`decay` or `lin_decay`] either to choose between the
    exponential or linear decay method
    """
    if epsilon_strategy == 'lin_decay':
        decayed_value = starting_value - ((starting_value - minimum_value) / (decay_step - 1)) * (global_step)
    else:
        decayed_value = starting_value * np.power(0.9, (global_step/decay_step))
    if decayed_value < minimum_value:
            return minimum_value
    return decayed_value

def parse_opt():
    """function to add support for command line parameters.
    `python q_lambda_mountain_car.py --help` for details.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--mass', type=float, default=PARAMS['mass'], help='the mass of the pole in kilograms')
    parser.add_argument('--friction', type=float, default=PARAMS['friction'], help='the mass of the cart in kilograms')
    parser.add_argument('--delta_t', type=float, default=PARAMS['delta_t'], help='time steps')
    parser.add_argument('--exploratory_strategy', default='suboptimal', choices=['suboptimal', 'adversarial'], help='exploratory policy strategy') 
    parser.add_argument('--alpha', type=float, default=PARAMS['alpha'], help='learning rate alpha')
    parser.add_argument('--lambda', type=float, default=PARAMS['lambda'], help='lambda value')
    parser.add_argument('--tot_episode', type=int, default=PARAMS['tot_episode'], help='number of episodes to run the simulation')
    parser.add_argument('--epsilon_strategy', default='decay', choices=['lin_decay', 'fixed', 'decay'], help='epsilon strategies to balance random exploration') 
    parser.add_argument('--epsilon', type=float, default=PARAMS['epsilon'], help='fixed epsilon value for fixed strategy')
    opt = parser.parse_args()
    return opt

def main(opt):
    # Initializing parameters from passed arguments
    mass, friction, delta_t, exploratory_strategy, alpha, lambda_, tot_episode, epsilon_strategy, epsilon_fixed = vars(opt).values()
    print("\nStarting simulation ...")
    print(vars(opt))

    # Directory to store experiment-specific results
    output_dir = create_output_dir(f'./outputs/Q_lambda_group4_{exploratory_strategy}_{lambda_}', alpha, tot_episode, epsilon_strategy, exploratory_strategy)

    # Initializing the environment with given parameters
    env = MountainCar(mass=mass, friction=friction, delta_t=delta_t)

    # Define the state arrays for velocity and position
    tot_action = 3  # Three possible actions
    tot_bins = 12  # the value used to discretize the space
    velocity_state_array = np.linspace(-1.5, +1.5, num=tot_bins-1, endpoint=False)
    position_state_array = np.linspace(-1.2, +0.5, num=tot_bins-1, endpoint=False)

    # Random policy as a square matrix of size (tot_bins x tot_bins)
    # Three possible actions represented by three integers
    policy_matrix = np.random.randint(low=0, high=tot_action, size=(tot_bins,tot_bins)).astype(np.float32)

    # Loading a suboptimal policy from SARSA 0
    if exploratory_strategy == 'adversarial':
        exploratory_policy_matrix = np.load('adversarial_policy.npy')
    else:
        exploratory_policy_matrix = np.load('exploratory_policy.npy')

    print("Policy Matrix:")
    print(policy_matrix)

    print("Exploratory Policy Matrix:")
    print_policy(exploratory_policy_matrix)

    # The state-action matrix and the visit counter
    # The rows are the velocities and the columns the positions.
    state_action_matrix = np.zeros((tot_action, tot_bins*tot_bins))
    visit_counter_matrix = np.zeros((tot_action, tot_bins*tot_bins))
    trace_matrix = np.zeros((tot_action, tot_bins*tot_bins))

    # Variables
    gamma = 0.999
    epsilon_start = 0.9  # those are the values for epsilon decay
    epsilon_stop = 0.1
    epsilon_decay_step = 3000
    if epsilon_strategy == 'lin_decay':
        epsilon_decay_step = tot_episode
    # print_episode = tot_episode//4  # print every...
    # movie_episode = tot_episode//4  # movie saved every...
    reward_list = list()
    step_list = list()
    eps_list = list() # to create epsilon curve for different E-greedy strategies
    first_success = None

    for episode in tqdm.tqdm(range(tot_episode)):
        if epsilon_strategy == "fixed": 
            epsilon = epsilon_fixed
        else:
            epsilon = return_decayed_value(epsilon_start, epsilon_stop, episode, decay_step=epsilon_decay_step, epsilon_strategy=epsilon_strategy)
        eps_list.append(epsilon)

        # Reset and return the first observation
        observation = env.reset(exploring_starts=True)
        # The observation is digitized, meaning that an integer corresponding
        # to the bin where the raw float belongs is obtained and use as replacement.
        observation = (np.digitize(observation[1], velocity_state_array),
                       np.digitize(observation[0], position_state_array))
        is_starting = True
        cumulated_reward = 0
        for step in range(100):
            #Take the action from the action matrix
            #action = policy_matrix[observation[0], observation[1]]
            #Take the action using epsilon-greedy
            action = return_epsilon_greedy_action(exploratory_policy_matrix, observation, epsilon=epsilon)
            if(is_starting):
                action = np.random.randint(0, tot_action)
                is_starting = False
            #Move one step in the environment and get obs and reward
            new_observation, reward, done = env.step(action)
            new_observation = (np.digitize(new_observation[1], velocity_state_array),
                               np.digitize(new_observation[0], position_state_array))
            new_action = policy_matrix[new_observation[0], new_observation[1]]
            #Updating the state-action matrix
            state_action_matrix = update_state_action(state_action_matrix, trace_matrix, visit_counter_matrix, observation, new_observation,
                   action, reward, alpha, gamma, lambda_, tot_bins)
            #Updating the policy
            policy_matrix = update_policy(policy_matrix, state_action_matrix, observation, tot_bins)
            #Increment the visit counter
            visit_counter_matrix = update_visit_counter(visit_counter_matrix, observation, action,
                                                        tot_bins)
            observation = new_observation
            cumulated_reward += reward
            if done: break

        # Store the data for statistics
        reward_list.append(cumulated_reward)
        step_list.append(step)
        if not first_success and step+1 < 100:
            print(step)
            first_success = episode
        # Printing utilities
        # if(episode % (print_episode-1) == 0):
        if episode+1 == tot_episode:
            print("")
            print("Episode: " + str(episode+1))
            print("Epsilon: " + str(epsilon))
            print("Episode steps: " + str(step+1))
            print("Cumulated Reward: " + str(cumulated_reward))
            # print("Policy matrix: ")
            # print_policy(policy_matrix)
        # if(episode % (movie_episode-1) == 0):
            print(f"Saving the reward plot in: {output_dir}/reward.png")
            plot_curve(reward_list, filepath=f"{output_dir}/reward.png",
                       x_label="Episode", y_label="Reward",
                       x_range=(0, len(reward_list)), y_range=(-1.1,1.1),
                       color="red", kernel_size=500,
                       alpha=0.4, grid=True, first_hundred=first_success)
            print(f"Saving the step plot in: {output_dir}/step.png")
            plot_curve(step_list, filepath=f"{output_dir}/step.png",
                       x_label="Episode", y_label="Steps",
                       x_range=(0, len(step_list)), y_range=(-0.1,100),
                       color="blue", kernel_size=500,
                       alpha=0.4, grid=True, first_hundred=first_success)
            print(f"Saving the gif in: {output_dir}/mountain_car.gif")
            env.render(file_path=f'{output_dir}/mountain_car.gif', mode='gif')
            print("Complete!")

            #Save the epsilon curve
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.set_xlabel("Episode")
            ax.set_ylabel("Epsilon")
            ax.plot(np.arange(len(eps_list)), eps_list, 'forestgreen')
            fig.savefig(f"{output_dir}/epsilon_Q_LAMBDA.png")
            fig.clear()
            plt.close(fig)

            print("Complete!")

    # Metrics for comparative analysis
    streak, start_i, end_i = calculate_longest_streak(step_list)
    with open(os.path.join(output_dir, 'metrics.txt'), 'w') as f:
        f.writelines(f'Q LAMBDA\nPARAMETERS: {vars(opt)}\n\
                    Mean Number of steps: {np.mean(step_list)}\n\
                    Median of steps: {np.median(step_list)}\n\
                    Max reward: {np.max(reward_list)}\n\
                    Cumulated reward: {cumulated_reward}\n\
                    Longest Streak of success: {streak} [{start_i}:{end_i}]\n\
                    Success ratio: {np.sum([1 for i in step_list if i+1 < 100]) / tot_episode} \n\
                    First Success: {first_success}')

    # Save reward and steps in npz file for later use
    # np.savez("./statistics.npz", reward=np.asarray(reward_list), step=np.asarray(step_list))
    # Time to check the utility matrix obtained
    print("Policy matrix after " + str(tot_episode) + " episodes:")
    print_policy(policy_matrix)
    np.save(f'{output_dir}/policy', policy_matrix)

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
