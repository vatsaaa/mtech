from abc import ABC, abstractmethod

class ISearchAlgorithm(ABC):
    @abstractmethod
    def search(self):
        pass

    def heuristic(self, row: int, col: int):
        pass
