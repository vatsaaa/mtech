import random

from Algorithms.ISearchAlgorithm import ISearchAlgorithm
from utils.GridEnvironment import GridEnvironment

class GeneticSearchAlgorithm(ISearchAlgorithm):
    population_size: int = 10
    mutation_rate: float = 0.01
    generations: int = 100

    # This class is needed only by GeneticSearchAlgorithm, so it as a nested class
    class Individual:
        def __init__(self, grid_env):
            individual_type = "Child" # Every individual is a child, until it is a parent

            path = [(grid_env.start[0], grid_env.start[1])]
            current_pos = grid_env.start
            while current_pos != grid_env.goal:
                adjacent_cells = grid_env.get_adjacent_cells(*current_pos)
                next_pos = random.choice(adjacent_cells)
                path.append(next_pos)
                current_pos = next_pos

            self.path = path
            self.fitness = 0

        def evaluate_fitness(self, grid_env):
            # Fitness function:
            # - Add 5 points for each step
            # - Deduct 5 points for each step near water bodies
            # - Deduct 3 points for each step on flooded roads
            for pos in self.path:
                row, col = pos
                if grid_env.grid[row][col] == '.':
                    self.fitness += 5
                elif grid_env.grid[row][col] == '#':
                    self.fitness -= 5
                elif grid_env.grid[row][col] == 'F':
                    self.fitness -= 3

    # TODO: Look carefully into this, crossover() actually makes a new parent
    def crossover(self, parent1, parent2):
        crossover_point = random.randint(1, min(len(parent1.path), len(parent2.path)) - 1)
        child_path = parent1.path[:crossover_point] + parent2.path[crossover_point:]
        return self.Individual(child_path)

    def mutate(self, individual, grid_env, mutation_rate):
        for i in range(1, len(individual.path) - 1):
            if random.random() < mutation_rate:
                adjacent_cells = grid_env.get_adjacent_cells(*individual.path[i])
                self.path[i] = random.choice(adjacent_cells)

    def search(self, grid_env: GridEnvironment):
        population = [self.Individual(grid_env) for _ in range(self.population_size)]

        for generation in range(self.generations):
            for individual in population:
                individual.evaluate_fitness(individual, grid_env)

            population.sort(key=lambda x: x.fitness, reverse=True)

            if population[0].fitness == 40:  # Max possible fitness - to be revised later 
                break

            next_generation = population[:2]  # Elitism
            while len(next_generation) < self.population_size:
                parent1 = random.choice(population[:self.population_size // 2])
                parent2 = random.choice(population[:self.population_size // 2])
                child = self.crossover(parent1, parent2)
                self.mutate(child, grid_env, self.mutation_rate)
                next_generation.append(child)

            population = next_generation

        best_individual = population[0]
        path = best_individual.path
        cost = 0
        for pos in path:
            row, col = pos
            if grid_env.grid[row][col] == '.':
                cost += 5
            elif grid_env.grid[row][col] == '#':
                cost -= 5
            elif grid_env.grid[row][col] == 'F':
                cost -= 3
        print("Final path:", path)
        print("Total cost:", cost)
        return path, cost

    def heuristic(self, row, col):
        path = []
        self._fitness(path)
        pass
