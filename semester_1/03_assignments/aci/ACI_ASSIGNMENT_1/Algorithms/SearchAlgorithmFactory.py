from abc import ABC

from Algorithms.GBFSearchAlgorithm import GBFSearchAlgorithm
from Algorithms.GeneticSearchAlgorithm import GeneticSearchAlgorithm

class SearchAlgorithmFactory:
    @staticmethod
    def create_search_algorithm(option):
        if option == "greedy":
            return GBFSearchAlgorithm()
        elif option == "genetic":
            return GeneticSearchAlgorithm()
        else:
            raise ValueError("Invalid search algorithm option")

