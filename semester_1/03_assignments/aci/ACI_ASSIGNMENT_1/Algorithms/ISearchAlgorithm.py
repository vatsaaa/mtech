from abc import ABC, abstractmethod

class ISearchAlgorithm(ABC):
    def __init__(self) -> None:
        super().__init__()
    
    @abstractmethod
    def search(self):
        pass

    def heuristic(self, row: int, col: int):
        pass
