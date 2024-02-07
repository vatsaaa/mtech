import random
from abc import ABC

from utils.GridEnvironment import GridEnvironment
from Algorithms.ISearchAlgorithm import ISearchAlgorithm

class Individual(ABC):
    def __init__(self, grid_env: GridEnvironment):
        self.individual_type = "Child"  # Every individual is a child, until it is a parent
        self.grid_env = grid_env
        print("Inside Individual __init__ -->>", grid_env.start)
        self.path = list(grid_env.start)
        print("Inside Individual __init__", self.path)
        self.fitness = 0

        current_pos = grid_env.start
        while current_pos != grid_env.goal:
            adjacent_cells = grid_env.get_adjacent_cells(*current_pos, algorithm="genetic")
            next_pos = random.choice(adjacent_cells)
            self.path.append(next_pos)
            current_pos = next_pos

    def evaluate_fitness(self):
        print("Inside evaluate_fitness......", self.path)
        for pos in self.path:
            print("Inside for loop of evaluate_fitness():", pos)
            row, col = pos
            if self.grid_env.grid[row][col] == '.':
                self.fitness += 5
            elif self.grid_env.grid[row][col] == '#':
                self.fitness -= 5
            elif self.grid_env.grid[row][col] == 'F':
                self.fitness -= 3

    def mutate(self):
        for i in range(1, len(self.path) - 1):
            if random.random() < self.mutation_rate:
                adjacent_cells = self.grid_env.get_adjacent_cells(*self.path[i], algorithm="genetic")
                self.path[i] = random.choice(adjacent_cells)

class GeneticSearchAlgorithm(ISearchAlgorithm):
    def __init__(self, grid_env):
        self.grid_env: GridEnvironment = grid_env
        self.population_size = 10
        self.generations = 100
        self.mutation_rate = 0.01

    def crossover(self, parent1, parent2):
        crossover_point = random.randint(1, min(len(parent1.path), len(parent2.path)) - 1)
        child_path = parent1.path[:crossover_point] + parent2.path[crossover_point:]
        return Individual(child_path)

    def search(self):
        population = [Individual(self.grid_env) for _ in range(self.population_size)]

        print("Inside search()")

        for generation in range(self.generations):
            print("Inside generation loop: ", generation)
            for individual in population:
                individual.evaluate_fitness()

            population.sort(key=lambda x: x.fitness, reverse=True)

            if population[0].fitness == 40:  # Max possible fitness - to be revised later 
                break

            next_generation = population[:2]  # Elitism
            while len(next_generation) < self.population_size:
                parent1 = random.choice(population[:self.population_size // 2])
                parent2 = random.choice(population[:self.population_size // 2])
                child = self.crossover(parent1, parent2)
                child.mutate(self.grid_env, self.mutation_rate)
                next_generation.append(child)

            population = next_generation

        best_individual = population[0]
        path = best_individual.path
        cost = 0
        for pos in path:
            row, col = pos
            if self.grid_env.grid[row][col] == '.':
                cost += 5
            elif self.grid_env.grid[row][col] == '#':
                cost -= 5
            elif self.grid_env.grid[row][col] == 'F':
                cost -= 3
        print("Final path:", path)
        print("Total cost:", cost)
        return path, cost
