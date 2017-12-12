import numpy as np
import gym
from gym import wrappers

max_episodes = 1000
learning_rate = 0.8
discount_factor = 0.95
max_iterations = 100
env_name = 'FrozenLake-v0'
env = gym.make(env_name)
q_table = np.zeros((env.observation_space.n, env.action_space.n))
rList = []
def train(render=False):
    reward100 = 0
    for ep in range(max_episodes):
        obs = env.reset()
        state = obs
        total_reward = 0
        for iter in range(max_iterations):
            if render:
                env.render()
            action = np.argmax(q_table[state,:] + np.random.randn(1,env.action_space.n)*(1./(ep+1)))
            obs, reward, done, info = env.step(action)
            total_reward += reward
            new_state = obs
            # update q table
            q_table[state,action] = q_table[state,action] + learning_rate * (reward + discount_factor *  np.max(q_table[new_state,:]) - q_table[state,action])
            state = new_state
            if done:
                break
        reward100 += total_reward
        if ep % 100 == 0:
            print('Iteration #{} -- Average reward = {}.'.format(ep+1, reward100/100))
            rList.append(reward100/100)
            if reward100/100 >= 0.78:
                print("Solved~!!!!")
                print("Average score every 100 trials: " +  str((rList)))
                print("Mean Score: {}".format(np.mean(rList)))
                break
            reward100 = 0

def run(render=True):
    obs = env.reset()
    total_reward = 0
    step_idx = 0
    for iter in range(max_iterations):
        if render:
            env.render()
        state = obs
        action = np.argmax(q_table[state])
        obs, reward, done, info = env.step(action)
        total_reward += discount_factor ** step_idx * reward
        step_idx += 1
        if done:
            break
    return total_reward

if __name__ == '__main__':
    train(render=False)
    solution_policy = np.argmax(q_table, axis=1)
    # run(render=True)
    print("Average score every 100 trials: " +  str((rList)))
    print("Mean Score: {}".format(np.mean(rList)))
