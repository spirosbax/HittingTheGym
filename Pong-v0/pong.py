import gym
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import RMSprop
from keras.layers import Input, Dense, Reshape
from keras.optimizers import Adam, Adamax, RMSprop
from keras.layers.core import Activation, Dropout, Flatten

# hyperparameters
H = 200 # number of hidden layer neurons
batch_size = 10 # every how many episodes to do a param update?
learning_rate = 1e-4
gamma = 0.99 # discount factor for reward
input_dim = 80 * 80 # input dimensionality: 80x80 grid
running_reward = None
env = gym.make("Pong-v0")
actions = env.action_space.n

render = True
resume = True # resume from previous checkpoint?
path = 'weights.h5' # path to save weights

def main():
    model = Sequential()
    model.add(Reshape((1,80,80), input_shape=(input_dim,)))
    model.add(Flatten())
    model.add(Dense(H, activation = 'relu', kernel_initializer='glorot_uniform'))
    model.add(Dense(actions, activation='softmax', kernel_initializer='glorot_uniform'))
    opt = RMSprop(lr=learning_rate)
    model.compile(loss='categorical_crossentropy', optimizer=opt)
    train(model)


def train(model):
    if resume:
        model.load_weights(path)
    images, probabilities, rewards = [],[],[]
    episode_number = 0
    episode_reward = 0
    previous_image = None
    running_reward = None
    obs = env.reset()
    while True:
        if render:
            env.render()

        current_image = preprocess(obs)
        # get the difference of 2 images, this way the model can detect movement
        image = current_image - previous_image if previous_image is not None else np.zeros(input_dim)
        previous_image = current_image

        # predict probabilities for every of the 6 possible actions (see https://ai.stackexchange.com/questions/2449/what-are-different-actions-in-action-space-of-environment-of-pong-v0-game-from)
        preds = model.predict(image.reshape([1, image.shape[0]]), batch_size=1).flatten()
        action = np.random.choice(actions, 1, p=preds) # choose actions

        # record various intermediates (needed later for backprop)
        images.append(image) # observation

        # grad that encourages the action that was taken to be taken (see http://cs231n.github.io/neural-networks-2/#losses if confused)
        y = np.zeros(actions)
        y[action] = 1
        probabilities.append(np.array(y).astype('float32') - preds)

        # obs is a 210*160*3 image
        obs, reward, done, info = env.step(action) # step the environment and get new measurements
        episode_reward += reward
        rewards.append(reward) # record reward (has to be done after we call step() to get reward for previous action)

        if done: # an episode finished
            episode_number += 1

            # stack together all inputs, hidden states, action gradients, and rewards for this episode
            episode_images = np.vstack(images)
            episode_probabilities = np.vstack(probabilities)
            episode_rewards = np.vstack(rewards)
            images, probabilities, rewards = [],[],[] # reset array memory

            # compute the discounted reward backwards through time
            discounted_ep_reward = discount_rewards(episode_rewards)

            # standardize the rewards to be unit normal (helps control the gradient estimator variance)
            discounted_ep_reward -= np.mean(discounted_ep_reward)
            discounted_ep_reward /= np.std(discounted_ep_reward)

            episode_probabilities *= discounted_ep_reward # modulate the gradient with advantage (PG magic happens right here.)
            grad = model.train_on_batch(episode_images, episode_probabilities)

            # boring book-keeping
            running_reward = episode_reward if running_reward is None else running_reward * 0.99 + episode_reward * 0.01
            print('resetting env. episode reward total was {}. running mean: {}'.format(episode_reward, running_reward))

            if episode_number % 100 == 0:
                model.save_weights("weights.h5")
            episode_reward = 0
            obs = env.reset() # reset env
            previous_image = None

        if reward != 0: # Pong has either +1 or -1 reward exactly when game ends.
            print("Episode {} Result: ".format(episode_number) + "Defeat!" if reward == -1 else "VICTORY!")

def preprocess(I):
    #preprocess 210x160x3 uint8 frame into 6400 (80x80) 1D float vector
    I = I[35:195] # crop
    I = I[::2,::2,0] # downsample by factor of 2
    I[I == 144] = 0 # erase background (background type 1)
    I[I == 109] = 0 # erase background (background type 2)
    I[I != 0] = 1 # everything else (paddles, ball) just set to 1
    return I.astype(np.float).ravel()

def discount_rewards(episode_rewards):
    # discount rewards
    running_add = 0
    discounted_rewards = np.zeros_like(episode_rewards)
    for i in reversed(range(0,episode_rewards.size)):
        if episode_rewards[i] != 0:
            running_add = 0
        running_add = running_add * gamma + episode_rewards[i]
        episode_rewards[i] = running_add
    return episode_rewards

if __name__ == "__main__":
    main()
