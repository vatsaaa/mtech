import argparse
from pprint import pprint

from utils.GridGenerator import GridGenerator
from utils.GridEnvironment import GridEnvironment
from Algorithms.GBFSearchAlgorithm import GBFSearchAlgorithm
from Algorithms.GeneticSearchAlgorithm import GeneticSearchAlgorithm
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

    if(args.genetic == False) and (args.gbfs == False):
        grid_gen = GridGenerator()
        grid_gen.generate_grid()

        env = GridEnvironment(grid_gen.grid, args.display)

        print("Running both Genetic Algorithm and Greedy Best First Search")
        gbfs = GBFSearchAlgorithm(env)
        gbfs_path, gbfs_cost = gbfs.search()

        print("Path taken by the agent using Greedy Best First Search:", gbfs_path)
        print("Total path cost using Greedy Best First Search:", gbfs_cost)

        genetic_search = GeneticSearchAlgorithm(env)
        gs_path, gs_cost = genetic_search.search()

        print("Path taken by the agent using Genetic Algorithm:", gs_path)
        print("Total path cost using Genetic Algorithm:", gs_cost)
    else:
        search_algorithm = SearchAlgorithmFactory.create_search_algorithm(args)
        path, total_cost = search_algorithm.search()
        print("Path taken by the agent:", path)
        print("Total path cost:", total_cost)

if __name__ == "__main__":
    main()
