import argparse
from pprint import pprint

## Import project libraries
from utils.grid import grid

from utils.GridEnvironment import GridEnvironment
from GBFS.gbfs import greedy_best_first_search
from GENETIC_ALGORITHM.geneticalgo import genetic_algorithm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--genetic", action="store_true", help="Run Genetic Algorithm")
    parser.add_argument("-b", "--gbfs", action="store_true", help="Run Greedy Best First Search")
    args = parser.parse_args()

    env = GridEnvironment(grid)
    pprint(env.grid) # Print the grid        

    if args.gbfs:
        print("Running Greedy Best First Search:")
        path, total_cost = greedy_best_first_search(env)
        print("Path taken by the agent:",path)
        print("Total path cost:", total_cost)
    elif args.genetic:
        print("\n\n\nRunning Genetic Algorithm:")
        env = GridEnvironment(grid)
        genetic_algorithm(env)
    else:
        print("Please select an algorithm to run. Use -h for help.")
