import retro
#import tensorflow as tf

def main():
    env = retro.make(game='Airstriker-Genesis',record='.', state='Level1')
    obs = env.reset()
    while True:
        obs, rew, done, info = env.step(env.action_space.sample())
        env.render()
        if done:
            break
            obs = env.reset()


if __name__ == '__main__':
    main()
