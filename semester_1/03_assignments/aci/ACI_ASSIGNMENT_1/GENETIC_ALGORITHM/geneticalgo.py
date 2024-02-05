import random

## Import project libraries
from utils.grid import grid

## Import project files
from utils.GridEnvironment import GridEnvironment

class Individual:
    def __init__(self, path):
        self.path = path
        self.fitness = 0

def create_individual(grid_env):
    path = [(grid_env.start[0], grid_env.start[1])]
    current_pos = grid_env.start
    while current_pos != grid_env.goal:
        adjacent_cells = grid_env.get_adjacent_cells(*current_pos)
        next_pos = random.choice(adjacent_cells)
        path.append(next_pos)
        current_pos = next_pos
    return Individual(path)

def evaluate_fitness(individual, grid_env):
    # Fitness function:
    # - Add 5 points for each step
    # - Deduct 5 points for each step near water bodies
    # - Deduct 3 points for each step on flooded roads
    fitness = 0
    for pos in individual.path:
        row, col = pos
        if grid_env.grid[row][col] == '.':
            fitness += 5
        elif grid_env.grid[row][col] == '#':
            fitness -= 5
        elif grid_env.grid[row][col] == 'F':
            fitness -= 3
    individual.fitness = fitness

def crossover(parent1, parent2):
    crossover_point = random.randint(1, min(len(parent1.path), len(parent2.path)) - 1)
    child_path = parent1.path[:crossover_point] + parent2.path[crossover_point:]
    return Individual(child_path)

def mutate(individual, grid_env, mutation_rate):
    for i in range(1, len(individual.path) - 1):
        if random.random() < mutation_rate:
            adjacent_cells = grid_env.get_adjacent_cells(*individual.path[i])
            individual.path[i] = random.choice(adjacent_cells)

def genetic_algorithm(grid_env, population_size=10, mutation_rate=0.01, generations=100):
    population = [create_individual(grid_env) for _ in range(population_size)]

    for generation in range(generations):
        for individual in population:
            evaluate_fitness(individual, grid_env)

        population.sort(key=lambda x: x.fitness, reverse=True)

        if population[0].fitness == 40:  # Max possible fitness - to be revised later 
            break

        next_generation = population[:2]  # Elitism
        while len(next_generation) < population_size:
            parent1 = random.choice(population[:population_size // 2])
            parent2 = random.choice(population[:population_size // 2])
            child = crossover(parent1, parent2)
            mutate(child, grid_env, mutation_rate)
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

env = GridEnvironment(grid)
genetic_algorithm(env)

