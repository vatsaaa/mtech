from abc import ABC
from pprint import pprint

from utils.grid import grid

from utils.GridEnvironment import GridEnvironment
from Algorithms.GBFSearchAlgorithm import GBFSearchAlgorithm
from Algorithms.GeneticSearchAlgorithm import GeneticSearchAlgorithm

class SearchAlgorithmFactory:
    @staticmethod
    def create_search_algorithm(args):
        env = GridEnvironment(grid)

        if args.display:
            pprint(env.grid) # Print the grid

        if args.gbfs:
            return GBFSearchAlgorithm(env)
        elif args.genetic:
            return GeneticSearchAlgorithm(env)
        else:
            print("Please select an algorithm to run. Use -h for help.")
            return
