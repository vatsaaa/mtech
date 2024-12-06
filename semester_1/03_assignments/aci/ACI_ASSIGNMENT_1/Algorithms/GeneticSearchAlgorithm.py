import random
from abc import ABC

from utils.grid import track_time_and_space
from utils.GridEnvironment import GridEnvironment
from Algorithms.ISearchAlgorithm import ISearchAlgorithm

class Individual(ABC):
    def __init__(self, grid_env: GridEnvironment):
        self.individual_type = "Child"  # Every individual is a child, until it is a parent
        self.grid_env = grid_env
        self.path = [tuple(grid_env.start)]
        self.fitness = 0

        current_pos = grid_env.start
        while current_pos != grid_env.goal:
            adjacent_cells = grid_env.get_adjacent_cells(*current_pos, algorithm="genetic")
            next_pos = random.choice(adjacent_cells)
            self.path.append(next_pos)
            current_pos = next_pos

    def set_path(self, path: list):
        self.path = path
     
    
    def evaluate_fitness(self):
        for pos in self.path:
            row, col = pos
            if self.grid_env.grid[row][col] == '.':
                self.fitness += 5
            elif self.grid_env.grid[row][col] == '#':
                self.fitness -= 5
            elif self.grid_env.grid[row][col] == 'F':
                self.fitness -= 3

    def mutate(self, mutation_rate: float):
        for i in range(1, len(self.path) - 1):
            if random.random() < mutation_rate:
                adjacent_cells = self.grid_env.get_adjacent_cells(*self.path[i], algorithm="genetic")
                self.path[i] = random.choice(adjacent_cells)

class GeneticSearchAlgorithm(ISearchAlgorithm):
    def __init__(self, grid_env: GridEnvironment):
        super().__init__(grid_env=grid_env)
        self.grid_env = grid_env
        
        self.population_size = 10
        self.generations = 100
        self.mutation_rate = 0.01

    def crossover(self, parent1, parent2):
        crossover_point = random.randint(1, min(len(parent1.path), len(parent2.path)) - 1)
        # Single point crossover
        child_path = parent1.path[:crossover_point] + parent2.path[crossover_point:]
        
        child = Individual(self.grid_env)
        child.set_path(child_path)
        
        return child

    @track_time_and_space
    def search(self):
        population = [Individual(self.grid_env) for _ in range(self.population_size)]
        self.total_nodes_expanded = 0
        for generation in range(self.generations):
            for individual in population:
                individual.evaluate_fitness()
                self.total_nodes_expanded += 1

            population.sort(key=lambda x: x.fitness, reverse=True)

            if population[0].fitness == 40:  # Max possible fitness - to be revised later 
                break

            next_generation = population[:2]  # Elitism
            while len(next_generation) < self.population_size:
                parent1 = random.choice(population[:self.population_size // 2])
                parent2 = random.choice(population[:self.population_size // 2])
                child = self.crossover(parent1, parent2)
                child.mutate(self.mutation_rate)
                next_generation.append(child)

            population = next_generation
        print("Memory Computation for Genetic Search is :", self.total_nodes_expanded)
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
        
        total_individuals = sum(len(individual.path) for individual in population)
        average_individual_size = total_individuals / self.population_size
        
        # Time complexity calculation: O(g * n * m)
        time_complexity = self.generations * self.population_size * average_individual_size
        print("Time Complexity of Genetic Search Algorithm:", time_complexity)
        space_complexity = self.population_size * len(population[0].path)
        print("Space Complexity of Genetic Search Algorithm:", space_complexity)

        return path, cost
