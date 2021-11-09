import numpy as np
from deap import base, creator, benchmarks

from deap import algorithms
from deap.tools._hypervolume import hv


import random
from deap import tools

# ne pas oublier d'initialiser la graine aléatoire (le mieux étant de le faire dans le main))
random.seed()

def my_nsga2(n, nbgen, evaluate, gym=False, ref_point=np.array([1,1]), IND_SIZE=5, weights=(-1.0, -1.0),details=True):
    """NSGA-2

    NSGA-2
    :param n: taille de la population
    :param nbgen: nombre de generation 
    :param evaluate: la fonction d'évaluation
    :param ref_point: le point de référence pour le calcul de l'hypervolume
    :param IND_SIZE: la taille d'un individu
    :param weights: les poids à utiliser pour la fitness (ici ce sera (-1.0,) pour une fonction à minimiser et (1.0,) pour une fonction à maximiser)
    """

    creator.create("MaFitness", base.Fitness, weights=weights)
    creator.create("Individual", list, fitness=creator.MaFitness)
    

    toolbox = base.Toolbox()
    paretofront = tools.ParetoFront()


    # à compléter
    toolbox = base.Toolbox()
    toolbox.register("attr_float", random.uniform,-5,5)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float,n=IND_SIZE)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("crossover",tools.cxSimulatedBinary,eta=15)
    toolbox.register("mutate",tools.mutPolynomialBounded,eta=15,low=-5,up=5,indpb=0.9)
    toolbox.register("evaluate",evaluate)
    toolbox.register("select",tools.selNSGA2)

    ## à compléter pour initialiser l'algorithme, n'oubliez pas de mettre à jour les statistiques, le logbook et le hall-of-fame.
    pop = toolbox.population(n)

    mean_x,mean_theta = [],[]

    fitnesses = map(toolbox.evaluate, pop)
    for ind, fit in zip(pop, fitnesses):
        #print(ind)
        ind.fitness.values = fit
    if gym :
        x,theta = 0,0
        for ind in pop :
            x += ind.fitness.values[0]
            theta += ind.fitness.values[1]
        mean_x.append(x/n)
        mean_theta.append(theta/n)

    paretofront.update(pop)     


    # Pour récupérer l'hypervolume, nous nous contenterons de mettre les différentes valeurs dans un vecteur s_hv qui sera renvoyé par la fonction.
    pointset=[np.array(ind.fitness.values) for ind in paretofront]
    #print((pointset))
    s_hv=[hv.hypervolume(pointset, ref_point)]

    # Begin the generational process
    for gen in range(1, nbgen):
        if details:
            if (gen%10==0):
                print("+",end="", flush=True)
            else:
                print(".",end="", flush=True)

        # Vary the population
        offspring = tools.selTournament(pop, int(len(pop)*0.5),tournsize=3)
        offspring = [toolbox.clone(ind) for ind in offspring]
        
        for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
            toolbox.crossover(ind1, ind2)
            toolbox.mutate(ind1)
            toolbox.mutate(ind2)
            del ind1.fitness.values, ind2.fitness.values
        
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        if gym :
            x,theta = 0,0
            for ind in pop :
                x += ind.fitness.values[0]
                theta += ind.fitness.values[1]
            mean_x.append(x/n)
            mean_theta.append(theta/n)
        # Select the next generation population
        pop = toolbox.select(pop + offspring, n)
        
        paretofront.update(pop)
        pointset=[np.array(ind.fitness.getValues()) for ind in paretofront]
        #print(np.array(pointset).shape)
        s_hv.append(hv.hypervolume(pointset, ref_point))
    
    if gym :
        return pop, paretofront, s_hv,mean_x,mean_theta
    else :
        return pop,paretofront,s_hv