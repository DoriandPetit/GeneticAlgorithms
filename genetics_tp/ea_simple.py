from matplotlib.pyplot import figure
import numpy as np
from deap import base, creator, benchmarks
import matplotlib.pyplot as plt

import random
from deap import tools

# ne pas oublier d'initialiser la graine aléatoire (le mieux étant de le faire dans le main))
random.seed()


def ea_simple(n, nbgen, evaluate, IND_SIZE, weights=(-1.0,),display=False,details=True):
    """Algorithme evolutionniste elitiste

    Algorithme evolutionniste elitiste. 
    :param n: taille de la population
    :param nbgen: nombre de generation 
    :param evaluate: la fonction d'évaluation
    :param IND_SIZE: la taille d'un individu
    :param weights: les poids à utiliser pour la fitness (ici ce sera (-1.0,) pour une fonction à minimiser et (1.0,) pour une fonction à maximiser)
    """
    lambda_select = int(n/2) #nombre d'individus selectionné : on prend la meilleur moitie
    k=0
    creator.create("MaFitness", base.Fitness, weights=weights)
    creator.create("Individual", list, fitness=creator.MaFitness)


    toolbox = base.Toolbox()

    toolbox.register("attribute", random.uniform,-5,5)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attribute, n=IND_SIZE)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("crossover", tools.cxSimulatedBinary, eta=15)
    toolbox.register("mutate", tools.mutPolynomialBounded, low=-5, up=5, eta=15.0, indpb=1/n)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("evaluate", evaluate)


    # Les statistiques permettant de récupérer les résultats
    stats = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    

    # La structure qui permet de stocker les statistiques
    logbook = tools.Logbook()
    logbook.header = "gen", "evals", "std", "min", "avg", "max"


    # La structure permettant de récupérer le meilleur individu
    hof = tools.HallOfFame(1)


    ## à compléter pour initialiser l'algorithme, n'oubliez pas de mettre à jour les statistiques, le logbook et le hall-of-fame.
    pop = toolbox.population(n)


    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    #print("invalid_ind, fitnesses",invalid_ind, fitnesses)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    # Compile statistics about the population
    record = stats.compile(pop)
    logbook.record(gen=0, evals=len(invalid_ind), **record)
    #print(logbook.stream)
    hof.update(pop)
    for g in range(1,nbgen):

        # Select the next generation individuals
        pop_select = toolbox.select(pop, lambda_select)
        offspring = pop_select
        offspring = list(map(toolbox.clone, offspring))
        
        # Apply crossover on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            toolbox.crossover(child1, child2)
            del child1.fitness.values
            del child2.fitness.values

        #Apply mutation
        for mutant in offspring:
            toolbox.mutate(mutant)
            del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in pop if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Compile statistics about the population
        record = stats.compile(pop)
        logbook.record(gen=g, evals=len(invalid_ind), **record)

        #pour conserver une taille constante on concatene les meilleur parents selectionné
        # et leur enfants
        pop = pop_select+offspring 

        # Pour voir l'avancement
        if details :
            if (g%10==0):
                print("+",end="", flush=True)
            else:
                print(".",end="", flush=True)

        if display :
            k+=1
            f_range = []
            f_pop = []
            x = np.linspace(-10,10,100)
            for i in np.arange(len(pop)):
                f_pop.append(evaluate(pop[i]))
            for i in x:
                f_range.append(evaluate([i]))
            if k % 10 == 0:
                plt.figure()
                #visualize the fonction ackley :
                plt.plot(x,f_range,c='red')
                #visualize population on the ackley slope : 
                plt.scatter(pop,f_pop,c='blue')
                plt.show()
                 
        hof.update(pop)

    return pop, hof, logbook

