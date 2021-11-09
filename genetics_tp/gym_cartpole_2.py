import cma
import gym
from deap import *
import numpy as np
from fixed_structure_nn_numpy import SimpleNeuralControllerNumpy

from deap import algorithms
from deap import base
from deap import benchmarks
from deap import creator
from deap import tools
import time
import array
import random

import math

from nsga2 import my_nsga2

nn=SimpleNeuralControllerNumpy(4,1,2,5)
IND_SIZE=len(nn.get_parameters())

env = gym.make('CartPole-v1')

def eval_nn(genotype, render=False, nbstep=500):
    total_x=0 # l'erreur en x est dans observation[0]
    total_theta=0 #  l'erreur en theta est dans obervation[2]
    total_reward=0
    x,theta=0,0
    nn=SimpleNeuralControllerNumpy(4,1,2,5)
    #print(np.array(genotype))
    nn.set_parameters(genotype)

    observation = env.reset()

    # à compléter
    for k in range(nbstep):
        if render: 
            env.render()
            time.sleep(0.1)
        action = nn.predict(observation)
        if action>0:
            action= 1
        else:
            action= 0

        observation, reward, done, info = env.step(action)
        total_reward+=reward
        if done:
            break
        #print(observation)
        total_x += np.abs(observation[0]) -x
        total_theta += np.abs(observation[2]) - theta
        x,theta = np.abs(observation[0]),np.abs(observation[2])
    # ATTENTION: vous êtes dans le cas d'une fitness à minimiser. 
    # Interrompre l'évaluation le plus rapidement possible est donc une stratégie que l'algorithme 
    # évolutionniste peut utiliser pour minimiser la fitness. Dans le cas ou le pendule tombe avant 
    # la fin, il faut donc ajouter à la fitness une valeur qui guidera l'apprentissage vers les bons 
    # comportements. Vous pouvez par exemple ajouter n fois une pénalité, n étant le nombre de pas de 
    # temps restant. Cela poussera l'algorithme à minimiser la pénalité et donc à éviter la chute. 
    # La pénalité peut être l'erreur au moment de la chute ou l'erreur maximale.
    total_x += (nbstep-total_reward) * np.abs(observation[0])-x
    total_theta += (nbstep-total_reward)*np.abs(observation[2]-theta)
    return (total_x, total_theta)



if (__name__ == "__main__"):

    pop, paretofront, s_hv = my_nsga2(100,100,eval_nn,IND_SIZE=IND_SIZE)
    eval_nn(paretofront[0],render=True)

    env.close()
