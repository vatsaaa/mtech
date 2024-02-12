from pprint import pprint

from utils.GridGenerator import GridGenerator
from utils.GridEnvironment import GridEnvironment
from Algorithms.GBFSearchAlgorithm import GBFSearchAlgorithm
from Algorithms.GeneticSearchAlgorithm import GeneticSearchAlgorithm

class SearchAlgorithmFactory:
    @staticmethod
    def create_search_algorithm(args, env: GridEnvironment):
        if not env:
            grid_gen = GridGenerator()
            grid_gen.generate_grid()
            env = GridEnvironment(grid_gen.grid, args.display)

        if args.display:
            pprint(env.grid) # Print the grid

        if args.gbfs:
            return GBFSearchAlgorithm(env)
        elif args.genetic:
            return GeneticSearchAlgorithm(env)
        else:
            print("Please select an algorithm to run. Use -h for help.")
            return
