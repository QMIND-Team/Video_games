
import retro
#from keras.model import Sequential
def main():
    env = retro.make(game='SonicTheHedgehog-Genesis')
    #model = Sequential()

    obs = env.reset()
    while True:

        obs, rew, done, info = env.step(env.action_space.sample())
        env.render()
        if done:
            break
            obs = env.reset()

        #env.render()
if __name__ == '__main__':

    main()

