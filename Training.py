import retro
import numpy as np
import cv2
import neat
import pickle
import time

env = retro.make(game="SonicTheHedgehog-Genesis", state="GreenHillZone.Act1", scenario='custom_senario')
imgarray = []
xpos_end = 0

resume = True
restore_file = "neat-checkpoint-449"



def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        ob = env.reset()
        ac = env.action_space.sample()

        inx, iny, inc = env.observation_space.shape

        inx = int(inx / 8)
        iny = int(iny / 8)

        net = neat.nn.recurrent.RecurrentNetwork.create(genome, config)

        current_max_fitness = 0
        fitness_current = 0
        frame = 0
        counter = 0
        xpos = 0

        start_time = time.time()

        done = False
        live=3
        while not done:



            env.render()
            frame += 1
            ob = cv2.resize(ob, (inx, iny))
            ob = cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY)
            ob = np.reshape(ob, (inx, iny))

            imgarray = np.ndarray.flatten(ob)

            nnOutput = net.activate(imgarray)

            ob, rew, done, info = env.step(nnOutput)
            xpos = info['x']
            rings = info['rings']
            seconds = (time.time() - start_time)

           

            # Sets the model fitness to the x-position plus rings, penalized for time
            # Change coefficients next to rings and seconds to change weightings
           
            fitness_current = (xpos + rings*100 - seconds*5)
           
            if xpos >= 65664:
                fitness_current += 10000000
                done = True

            # Counter function to time out when progress is not made
            if fitness_current > current_max_fitness:
                current_max_fitness = fitness_current
                counter = 0
            else:
                counter += 1

            if done or counter == 300:#
                done = True
                print(genome_id, fitness_current)
                print("Time: ", seconds)
                start_time = time.time()

            genome.fitness = fitness_current


config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     'config-feedforward')

if resume == True:
    p = neat.Checkpointer.restore_checkpoint(restore_file)
else:
    p = neat.Population(config)

p.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
p.add_reporter(stats)
p.add_reporter(neat.Checkpointer(1))

winner = p.run(eval_genomes)

with open('winner.pkl', 'wb') as output:
    pickle.dump(winner, output, 1)