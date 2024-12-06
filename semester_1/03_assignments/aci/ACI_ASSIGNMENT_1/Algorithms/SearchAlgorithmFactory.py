from pprint import pprint

from utils.GridGenerator import GridGenerator
from utils.GridEnvironment import GridEnvironment
from Algorithms.GBFSearchAlgorithm import GBFSearchAlgorithm
from Algorithms.GeneticSearchAlgorithm import GeneticSearchAlgorithm

class SearchAlgorithmFactory:
    @staticmethod
    def create_search_algorithm(args, env: GridEnvironment):
        display = True if "-d" in args or "--display" in args else False
        if not env:
            grid_gen = GridGenerator()
            grid_gen.generate_grid()
            env = GridEnvironment(grid_gen.grid, display)

        if display:
            pprint(env.grid) # Print the grid

        gbfs = True if "-b" in args or "--gbfs" in args else False
        genetic = True if "-a" in args or "--genetic" in args else False
        if gbfs:
            return GBFSearchAlgorithm(env)
        elif genetic:
            return GeneticSearchAlgorithm(env)
        else:
            print("Please select an algorithm to run. Use -h for help.")
            return
