import gym
import random
import math
import numpy as np
import time

#Creating environment
env_to_use = 'CartPole-v0'
env = gym.make(env_to_use)

"""Defining some constants"""

MAX_EPISODES = 1000
MAX_TIMESTAMPS = 100000
STREAK_TO_END = 100
SOLVED = 194
RENDER = False
DEBUG_MODE = True

#Setting dimensions for each state dimension
NUM_BUCKETS = (1, 1, 6, 3)
NUM_ACTIONS = env.action_space.n

STATE_BOUNDS = list(zip(env.observation_space.low, env.observation_space.high))
STATE_BOUNDS[1] = [-0.5, 0.5]
STATE_BOUNDS[3] = [-math.radians(50), math.radians(50)]

# Index of the action
ACTION_INDEX = len(NUM_BUCKETS)

## Creating a Q-Table for each state-action pair
q_table = np.zeros(NUM_BUCKETS + (NUM_ACTIONS,))

## Learning related constants
MIN_EXPLORATION_RATE = 0.01
MIN_LEARNING_RATE = 0.1

def main():
    t = time.time()
    train()
    print("Solved after {} seconds".format(time.time() - t))
    run(True)


def train(render=False):
    #Initializing learning and exploring rates
    learning_rate = get_learning_rate(0)
    print(learning_rate)
    exploration_rate = get_exploration_rate(0)
    discount_factor = 0.99  # since the world is unchanging

    num_streaks = 0
    for ep in range(MAX_EPISODES):
        observations = env.reset()
        previous_state = bucketize(observations)
        for t in range(MAX_TIMESTAMPS):
            if render:
                env.render()

            #select best action at this state
            action = select_best_action(previous_state, exploration_rate)

            #Execute action
            observations, reward, done, info = env.step(action)

            #Get current state
            current_state = bucketize(observations)

            #Update previous_state Q-Table value
            best_reward = np.amax(q_table[current_state])
            q_table[previous_state + (action,)] += learning_rate * (reward + discount_factor * (best_reward) - q_table[previous_state + (action,)])

            # Setting up for the next iteration
            previous_state = current_state

            if done:
                if t >= SOLVED:
                    num_streaks += 1
                else:
                    num_streaks = 0

            if (DEBUG_MODE):
                print("\nEpisode = %d" % ep)
                print("t = %d" % t)
                print("Action: %d" % action)
                print("State: %s" % str(current_state))
                print("Reward: %f" % reward)
                print("Best Q: %f" % best_reward)
                print("Explore rate: %f" % exploration_rate)
                print("Learning rate: %f" % learning_rate)
                print("Streaks: %d" % num_streaks)
                print("")

            if done:
                print("Episode {} finished after {} timestamps with {} streaks".format(ep, t, num_streaks))
                break

        # It's considered done when it's solved over 100 times consecutively
        if num_streaks >= STREAK_TO_END:
            break

        # Update parameters
        exploration_rate = get_exploration_rate(ep)
        learning_rate = get_learning_rate(ep)

def select_best_action(state, exploration_rate):
    #This part enables exploring new actions that maybe don't offer that immidiate highest reward
    if random.random() < exploration_rate:
        action = env.action_space.sample()
    else:
        # Select the action with the highest reward
        action = np.argmax(q_table[state])
    return action

def bucketize(state):
    bucket_indice = []
    for i in range(len(state)):
        if state[i] <= STATE_BOUNDS[i][0]:
            bucket_index = 0
        elif state[i] >= STATE_BOUNDS[i][1]:
            bucket_index = NUM_BUCKETS[i] - 1
        else:
            # Mapping the state bounds to the bucket array
            bound_width = STATE_BOUNDS[i][1] - STATE_BOUNDS[i][0]
            offset = (NUM_BUCKETS[i]-1)*STATE_BOUNDS[i][0]/bound_width
            scaling = (NUM_BUCKETS[i]-1)/bound_width
            bucket_index = int(round(scaling*state[i] - offset))
        bucket_indice.append(bucket_index)
    return tuple(bucket_indice)

def get_exploration_rate(t):
    return max(MIN_EXPLORATION_RATE, min(1, 1.0 - math.log10((t+1)/25)))

def get_learning_rate(t):
    return max(MIN_LEARNING_RATE, min(0.5, 1.0 - math.log10((t+1)/25)))

def run(render=False):
        observations = env.reset()
        state = bucketize(observations)
        while True:
            env.render()
            #select best action at this state
            action = select_best_action(state, 0)
            #Execute action
            observations, reward, done, info = env.step(action)
            #Get current state
            state = bucketize(observations)
            if done:
                break

if __name__ == "__main__":
    main()
