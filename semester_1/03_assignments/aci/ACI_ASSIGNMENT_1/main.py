import argparse
from pprint import pprint

## Import project libraries
from utils.grid import grid

from utils.GridEnvironment import GridEnvironment
from Algorithms.SearchAlgorithmFactory import SearchAlgorithmFactory

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-a", "--genetic", action="store_true", help="Run Genetic Algorithm")
    group.add_argument("-b", "--gbfs", action="store_true", help="Run Greedy Best First Search")
    parser.add_argument("-d", "--display", action="store_true", help="Display grid")
    args = parser.parse_args()

    env = GridEnvironment(grid)

    if args.display and (args.genetic or args.gbfs):
        pprint(env.grid) # Print the grid

    if args.gbfs:
        option = "greedy"
    elif args.genetic:
        option = "genetic"
    else:
        print("Please select an algorithm to run. Use -h for help.")
    
    search_algorithm = SearchAlgorithmFactory.create_search_algorithm(option)
    path, total_cost = search_algorithm.search(env)

    print("Path taken by the agent:",path)
    print("Total path cost:", total_cost)
