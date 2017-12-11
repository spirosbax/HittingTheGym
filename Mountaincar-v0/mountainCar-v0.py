import numpy as np
import gym
from gym import wrappers

n_states = 40
max_episodes = 10000
initial_lr = 1.0 #Initial Learning rate
min_lr = 0.003
discount_factor = 1.0
max_iterations = 10000
eps = 0.02
env_name = 'MountainCar-v0'
env = gym.make(env_name)
env.seed(0)
np.random.seed(0)
q_table = np.zeros((n_states, n_states, env.action_space.n))

def train(render=False):
    for ep in range(max_episodes):
        obs = env.reset()
        total_reward = 0
        ## eta: learning rate is decreased at each step
        eta = max(min_lr, initial_lr * (0.85 ** (ep//100)))
        for iter in range(max_iterations):
            if render:
                env.render()
            state = obs_to_state(obs)
            if np.random.uniform(0, 1) < eps:
                action = np.random.choice(env.action_space.n)
            else:
                logits = q_table[state]
                logits_exp = np.exp(logits)
                probs = logits_exp / np.sum(logits_exp)
                action = np.random.choice(env.action_space.n, p=probs)
            obs, reward, done, info = env.step(action)
            total_reward += reward
            # update q table
            new_state = obs_to_state(obs)
            q_table[state + (action,)] = q_table[state + (action,)] + eta * (reward + discount_factor *  np.max(q_table[new_state]) - q_table[state + (action, )])
            if done:
                break
        if ep % 100 == 0:
            print('Iteration #{} -- Total reward = {}.'.format(ep+1, total_reward))

def run(render=True, policy=None):
    obs = env.reset()
    total_reward = 0
    step_idx = 0
    for iter in range(max_iterations):
        if render:
            env.render()
        if policy is None:
            action = env.action_space.sample()
        else:
            state =  obs_to_state(obs)
            action = policy[state]
        obs, reward, done, info = env.step(action)
        total_reward += discount_factor ** step_idx * reward
        step_idx += 1
        if done:
            break
    return total_reward

def obs_to_state(obs):
    """ Maps an observation to state """
    env_low = env.observation_space.low
    env_high = env.observation_space.high
    env_dx = (env_high - env_low) / n_states
    a = int((obs[0] - env_low[0])/env_dx[0])
    b = int((obs[1] - env_low[1])/env_dx[1])
    return a, b

if __name__ == '__main__':
    train(render=False)
    solution_policy = np.argmax(q_table, axis=2)

    # Animate it
    solution_policy_scores = [run(render=False, policy=solution_policy) for _ in range(100)]
    print("Average score of solution = ", np.mean(solution_policy_scores))
    run(render=True, policy=solution_policy)
