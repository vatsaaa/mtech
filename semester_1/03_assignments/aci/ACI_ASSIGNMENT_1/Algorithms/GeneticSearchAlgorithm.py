from Algorithms.ISearchAlgorithm import ISearchAlgorithm
from utils.GridEnvironment import GridEnvironment

class GeneticSearchAlgorithm(ISearchAlgorithm):
    def search(self, grid_env: GridEnvironment):
        pass

    def _fitness(self, path):
        pass

    def heuristic(self, row, col):
        path = []
        self._fitness(path)
        pass
