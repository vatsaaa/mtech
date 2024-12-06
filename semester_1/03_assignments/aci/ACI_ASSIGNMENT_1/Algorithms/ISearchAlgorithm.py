from abc import ABC, abstractmethod
from utils.GridEnvironment import GridEnvironment

class ISearchAlgorithm(ABC):
    def __init__(self, grid_env: GridEnvironment) -> None:
        super().__init__()
    
    @abstractmethod
    def search(self):
        pass

    def heuristic(self, row: int, col: int):
        pass
