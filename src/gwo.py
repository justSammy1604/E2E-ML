import numpy as np
import random
import copy
import sys

class GreyWolfOptimizer:
    class Wolf:
        def __init__(self, fitness, dim, minx, maxx, seed):
            self.rnd = np.random.Random(seed)
            self.position = np.array([0.0] * dim)
            for i in range(dim):
                self.position[i] = ((maxx - minx) * self.rnd.rand() + minx)
            self.fitness = fitness(self.position)

    def __init__(self, fitness_func, dim, pop_size, max_iter, minx=0, maxx=1, binary=True):
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
            bin_position = self.binarize(position)
            return self.fitness_func(bin_position)
        else:
            return self.fitness_func(position)

    def optimize(self):
        rnd = np.random.Random(0)
        population = [self.Wolf(self.get_fitness, self.dim, self.minx, self.maxx, i) for i in range(self.pop_size)]
        population = sorted(population, key=lambda w: w.fitness)
        alpha, beta, gamma = copy.copy(population[:3])

        Iter=0
        while Iter < self.max_iter:
            a = 2 * (1 - Iter / self.max_iter)
            for i in range(self.pop_size):
                A1, A2, A3 = (
                    a * (2 * rnd.random() - 1),
                    a * (2 * rnd.random() - 1),
                    a * (2 * rnd.random() - 1),
                )
                C1, C2, C3 = 2 * rnd.random(), 2 * rnd.random(), 2 * rnd.random()

                X1 = alpha.position - A1 * np.abs(
                    C1 * alpha.position - population[i].position
                )
                X2 = beta.position - A2 * np.abs(
                    C2 * beta.position - population[i].position
                )
                X3 = gamma.position - A3 * np.abs(
                    C3 * gamma.position - population[i].position
                )
                Xnew = (X1 + X2 + X3) / 3.0

                fnew = self.get_fitness(Xnew)
                if fnew < population[i].fitness:
                    population[i].position = Xnew
                    population[i].fitness = fnew

            population = sorted(population, key=lambda w: w.fitness)
            alpha, beta, gamma = copy.copy(population[:3])
            Iter += 1
        
        return alpha.position, alpha.fitness
