import numpy as np
import random
import copy
import sys

class FireflyAlgorithm:

    def __init__(
        self, fitness_func, dim, pop_size, max_iter, minx=0, maxx=1, binary=True
    ):
        self.fitness_func = fitness_func
        self.dim = dim
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.minx = minx
        self.maxx = maxx
        self.binary = binary

    def binarize(self, position):
        sigmoid = 1 / (1 + np.exp(-position))
        return np.where(np.random.rand(self.dim) < sigmoid, 1, 0)

    def get_fitness(self, position):
        if self.binary:
            bin_pos = self.binarize(position)
            return self.fitness_func(bin_pos)
        else:
            return self.fitness_func(position)

    def optimize(self):
        population = self.minx + (self.maxx - self.minx) * np.random.rand(
            self.pop_size, self.dim
        )
        fitness = np.array([self.get_fitness(pos) for pos in population])

        for iteration in range(self.max_iter):
            alpha = 0.2
            beta = 1
            gamma = 1

            for i in range(self.pop_size):
                for j in range(self.pop_size):
                    if fitness[j] < fitness[i]:
                        distance = np.linalg.norm(population[i] - population[j])
                        attractiveness = np.exp(-gamma * distance**2)
                        population[i] += alpha * attractiveness * (
                            population[j] - population[i]
                        ) + beta * (np.random.rand(self.dim) - 0.5)

                population[i] = np.clip(population[i], self.minx, self.maxx)

            fitness = np.array([self.get_fitness(pos) for pos in population])

        best_index = np.argmin(fitness)
        best_position = population[best_index]
        best_fitness = fitness[best_index]

        return best_position, best_fitness
