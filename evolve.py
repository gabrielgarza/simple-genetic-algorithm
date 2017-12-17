import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import random

class Individual(object):

    def __init__(self, pixels=None, mutate_prob=0.01):
        if pixels is None:
            self.pixels = np.random.randint(101, size=5)
        else:
            self.pixels = pixels
            # Mutate
            if mutate_prob > np.random.rand():
                mutate_index = np.random.randint(len(self.pixels) - 1)
                self.pixels[mutate_index] = np.random.randint(101)


    def fitness(self):
        """
            Returns fitness of individual
            Fitness is the difference between
        """
        target_sum = 500
        return abs(target_sum - np.sum(self.pixels))

class Population(object):

    def __init__(self, pop_size=10, mutate_prob=0.01, retain=0.2, random_retain=0.03):
        """
            Args
                pop_size: size of population
                fitness_goal: goal that population will be graded against
        """
        self.pop_size = pop_size
        self.mutate_prob = mutate_prob
        self.retain = retain
        self.random_retain = random_retain
        self.fitness_history = []
        self.parents = []

        # Create individuals
        self.individuals = []
        for x in range(pop_size):
            self.individuals.append(Individual(pixels=None,mutate_prob=self.mutate_prob))

    def grade(self, episode=None):
        fitness_sum = 0
        for x in self.individuals:
            fitness_sum += x.fitness()

        pop_fitness = fitness_sum / self.pop_size
        self.fitness_history.append(pop_fitness)
        if episode is not None:
            if episode % 500 == 0:
                print("Episode",episode,"Population fitness:", pop_fitness)


    def select_parents(self):
        # Sort individuals by fitness
        self.individuals = list(reversed(sorted(self.individuals, key=lambda x: x.fitness(), reverse=True)))
        # Keep the fittest as parents for next gen
        retain_length = self.retain * len(self.individuals)
        self.parents = self.individuals[:int(retain_length)]

        # Randomly select some from unfittest and add to parents array
        unfittest = self.individuals[int(retain_length):]
        for unfit in unfittest:
            if self.random_retain > np.random.rand():
                self.parents.append(unfit)

    def breed(self):
        target_children_size = self.pop_size - len(self.parents)
        children = []
        while len(children) < target_children_size:
            father = random.choice(self.parents)
            mother = random.choice(self.parents)
            if father != mother:
                child_pixels = [ random.choice(pixel_pair) for pixel_pair in zip(father.pixels, mother.pixels)]
                child = Individual(child_pixels)
                children.append(child)
        self.individuals = self.parents + children

    def evolve(self):
        self.grade()
        self.select_parents()
        self.breed()
        # Reset parents and children
        self.parents = []
        self.children = []

if __name__ == "__main__":
    pop = Population(pop_size=100, mutate_prob=0.01, retain=0.2, random_retain=0.03)
    pop.grade()

    EPISODES = 5000
    for x in range(EPISODES):
        pop.evolve()
        pop.grade(episode=x)

    for i in pop.individuals:
        print(i.pixels)

    # Plot fitness history
    matplotlib.use("MacOSX")
    plt.plot(np.arange(len(population.fitness_history)), population.fitness_history)
    plt.ylabel('Fitness')
    plt.xlabel('Episodes')
    plt.show()
