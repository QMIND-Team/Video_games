import sys
# sys.path.append("C:/Users/Oliver/Anaconda3/envs/gym/Lib/site-packages")
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

ENV_NAME = 'StreetFighterIISpecialChampionEdition-Genesis'

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

    dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10,
                   target_model_update=1e-2, policy=policy)
    dqn.compile(Adam(lr=1e-3), metrics=['mae'])

    training_history = dqn.fit(env, nb_steps=50000, visualize=True, verbose=2)
    e = training_history.history['episode_reward']
    plt.plot(episode_reward)
    plt.ylabel('episode reward')
    plt.show()
    dqn.save_weights('dqn_{}_weights.h5f'.format(ENV_NAME), overwrite=True)

    dqn.test(env, nb_episodes=5, visualize=True)


if __name__ == "__main__":
    main()
