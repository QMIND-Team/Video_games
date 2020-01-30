import sys
#sys.path.append("C:/Users/Oliver/Anaconda3/envs/gym/Lib/site-packages")
#sys.path.append("O:\Oliver\Anaconda\envs\gym\Lib\site-packages")
import numpy as np
import gym
import retro
import h5py
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

ENV_NAME = 'SonicTheHedgehog-Genesis'

def main():
    env = retro.make(game=ENV_NAME, use_restricted_actions=retro.Actions.DISCRETE)
    nb_actions = env.action_space.n
    model = Sequential()
    model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dense(nb_actions))
    model.add(Activation('linear'))

    memory = SequentialMemory(limit=50000, window_length=1)
    policy = BoltzmannQPolicy()

    # Uncomment the following line to load the model weights from file
    # NOTE: on the first run you must comment the line below (38) since there wont be
    # any weights to load and you will get an error
    model.load_weights('dqn_{}_weights.h5f'.format(ENV_NAME))
    dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=100,
                   target_model_update=1e-3, policy=policy)
    dqn.compile(Adam(lr=1e-3), metrics=['mae'])
    training_history = dqn.fit(env, nb_steps=2000, visualize=True, verbose=2, action_repetition=4)
    # Threw an error, I don't think we need to graph things...
    # plot_training_results(training_history)

    # Uncomment the following line to overwrite the model weights file after training
    # dqn.save_weights('dqn_{}_weights.h5f'.format(ENV_NAME), overwrite=True)
    dqn.test(env, nb_episodes=2, visualize=True)


def plot_training_results(training_history):
    session_reward = np.array(training_history.history['episode_reward'])
    session_episodes = np.arange(session_reward.size)

    overall_reward = np.load('reward_history.npy')
    overall_reward = np.concatenate((overall_reward, session_reward))
    np.save('reward_history.npy', session_reward)  # save

    session_regression_line = calculate_regression_line(session_episodes, session_reward)

    plt.scatter(session_episodes, session_reward)
    plt.plot(session_episodes, session_regression_line)
    plt.title('training session results')
    plt.ylabel('episode reward')
    plt.show()

    overall_episodes = np.arange(overall_reward.size)
    overall_regression_line = calculate_regression_line(overall_episodes, overall_reward)

    plt.scatter(overall_episodes, overall_reward)
    plt.plot(overall_episodes, overall_regression_line)
    plt.title('overall training results')
    plt.ylabel('episode reward')
    plt.show()


def calculate_regression_line(episodes, rewards):
    slope = (((np.mean(episodes) * np.mean(rewards)) - np.mean(episodes * rewards)) /
         ((np.mean(episodes) * np.mean(episodes)) - np.mean(episodes * episodes)))
    intercept = np.mean(rewards) - slope * np.mean(episodes)
    regression_line = (slope * episodes) + intercept
    return regression_line


if __name__ == "__main__":
    main()
