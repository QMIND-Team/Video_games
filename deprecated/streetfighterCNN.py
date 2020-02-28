import sys
sys.path.append("O:\Oliver\Anaconda\envs\gym\Lib\site-packages")
import retro
import h5py
from CNNProcessor import CNNProcessor
from InfoCallbackTrain import InfoCallbackTrain
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Conv2D
from keras.optimizers import Adam
import os.path
from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory
from trainingMetrics import plot_reward, plot_wins, STATE_NAME

ENV_NAME = 'StreetFighterIISpecialChampionEdition-Genesis'

def main():
    env = retro.make(game=ENV_NAME, state=STATE_NAME, use_restricted_actions=retro.Actions.DISCRETE)
    nb_actions = env.action_space.n
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(8, 8), strides=4, activation="relu", input_shape=(1,) + (128, 100), data_format='channels_first'))
    model.add(Conv2D(64, kernel_size=(4, 4), strides=2, activation="relu"))
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(Flatten())
    model.add(Dense(512, activation="relu"))
    model.add(Dense(nb_actions))
    model.add(Activation('linear'))
    memory = SequentialMemory(limit=50000, window_length=1)
    policy = BoltzmannQPolicy()
    print(env.observation_space)

    # Uncomment the following line to load the model weights from file
    if os.path.exists('./weights/dqn_cnn_{}_weights.h5f'.format(STATE_NAME)):
        model.load_weights('./weights/dqn_cnn_{}_weights.h5f'.format(STATE_NAME))
    dqn = DQNAgent(processor=CNNProcessor(), model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10000,
               target_model_update=1e-3, policy=policy)
    dqn.compile(Adam(lr=1e-3), metrics=['mae'])

    dqn.fit(env, nb_steps=1000000, visualize=True, verbose=2, callbacks=[InfoCallbackTrain()], action_repetition=4)
    dqn.save_weights('./weights/dqn_cnn_{}_weights.h5f'.format(STATE_NAME), overwrite=True)
    plot_wins()
    #plot_reward(training_history)

    # Uncomment the following line to overwrite the model weights file after training

    dqn.test(env, nb_episodes=5, visualize=True)


if __name__ == "__main__":
    main()
