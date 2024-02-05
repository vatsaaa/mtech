import argparse
from pprint import pprint

from utils.grid import grid
from utils.GridEnvironment import GridEnvironment
from Algorithms.SearchAlgorithmFactory import SearchAlgorithmFactory

def main():
    """
    This script runs a search algorithm on a grid environment.

    Usage:
        python main.py [-a | -b] [-d]

    Options:
        -a, --genetic   Run Genetic Algorithm
        -b, --gbfs      Run Greedy Best First Search
        -d, --display   Display grid

    Example:
        python main.py -b -d
    """

    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-a", "--genetic", action="store_true", help="Run Genetic Algorithm")
    group.add_argument("-b", "--gbfs", action="store_true", help="Run Greedy Best First Search")
    parser.add_argument("-d", "--display", action="store_true", help="Display grid")
    args = parser.parse_args()

    env = GridEnvironment(grid)

    if args.display:
        pprint(env.grid) # Print the grid

    if args.gbfs:
        algorithm = "greedy"
    elif args.genetic:
        algorithm = "genetic"
    else:
        print("Please select an algorithm to run. Use -h for help.")
        return
    
    search_algorithm = SearchAlgorithmFactory.create_search_algorithm(algorithm)
    path, total_cost = search_algorithm.search(env)

    print("Path taken by the agent:", path)
    print("Total path cost:", total_cost)

if __name__ == "__main__":
    main()
