import numpy as np
import matplotlib
import matplotlib.pyplot as plt

class Individual(object):

    def __init__(self, arg):
        self.arg = arg

    def fitness(self):


class Population(object):

    def __init__(self, pop_size, fitness_goal):
        """
            Args
                pop_size: size of population
                fitness_goal: goal that population will be graded against
        """
        self.fitness_history = []

        # Create individuals
        self.individuals = individual() for x in xrange(pop_size)

    def grade(self):

        self.fitness_history.append(grade)

    def mutate(self):

    def evolve(self):



if __name__ == "__main__":
    population = Population(pop_size=100, target=500)
    population.grade()

    EPISODES = 100
    for x in xrange(EPISODES):
        population.evolve()
        population.grade()

    # Plot fitness history

    matplotlib.use("MacOSX")
    plt.plot(np.arange(len(population.fitness_history)), population.fitness_history)
    plt.ylabel('Fitness')
    plt.xlabel('Episodes')
    plt.show()
