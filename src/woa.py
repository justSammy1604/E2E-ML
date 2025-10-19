import numpy as np
import random
import math
import copy
import sys

class WhaleOptimizationAlgorithm:
    class Whale:
        def __init__(self, fitness, dim, minx, maxx, seed):
            self.rnd = random.Random(seed)
            self.position = np.array([0.0] * dim)
            for i in range(dim):
                self.position[i] = (maxx - minx) * self.rnd.random() + minx
            self.fitness = fitness(self.position)

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
        rnd = random.Random(0)
        population = [
            self.Whale(self.get_fitness, self.dim, self.minx, self.maxx, i)
            for i in range(self.pop_size)
        ]

        Fbest = sys.float_info.max
        Xbest = np.zeros(self.dim)
        for i in range(self.pop_size):
            if population[i].fitness < Fbest:
                Fbest = population[i].fitness
                Xbest = np.copy(population[i].position)

        Iter = 0
        while Iter < self.max_iter:
            a = 2 * (1 - Iter / self.max_iter)
            a2 = -1 + Iter * ((-1) / self.max_iter)
            for i in range(self.pop_size):
                A = 2 * a * rnd.random() - a
                C = 2 * rnd.random()
                b = 1
                l = (a2 - 1) * rnd.random() + 1
                p = rnd.random()

                if p < 0.5:
                    if abs(A) < 1:
                        D = np.abs(C * Xbest - population[i].position)
                        Xnew = Xbest - A * D
                    else:
                        rand_idx = random.randint(0, self.pop_size - 1)
                        while rand_idx == i:
                            rand_idx = random.randint(0, self.pop_size - 1)
                        Xrand = population[rand_idx].position
                        D = np.abs(C * Xrand - population[i].position)
                        Xnew = Xrand - A * D
                else:
                    D1 = np.abs(Xbest - population[i].position)
                    Xnew = D1 * math.exp(b * l) * math.cos(2 * math.pi * l) + Xbest

                population[i].position = Xnew

            for i in range(self.pop_size):
                population[i].position = np.clip(
                    population[i].position, self.minx, self.maxx
                )
                population[i].fitness = self.get_fitness(population[i].position)
                if population[i].fitness < Fbest:
                    Xbest = np.copy(population[i].position)
                    Fbest = population[i].fitness

            Iter += 1

        return Xbest, Fbest
