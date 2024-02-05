from pprint import pprint

## Import project libraries
from utils.grid import grid

from utils.GridEnvironment import GridEnvironment
from GBFS.gbfs import greedy_best_first_search
from GENETIC_ALGORITHM.geneticalgo import genetic_algorithm

if __name__ == "__main__":
    print("Running Greedy Best First Search:")
    env = GridEnvironment(grid)
    pprint(env.grid) # Print the grid
    
    path, total_cost = greedy_best_first_search(env)
    print("Path taken by the agent:",path)
    print("Total path cost:", total_cost)

    print("\n\n\nRunning Genetic Algorithm:")
    genetic_algorithm(env)
