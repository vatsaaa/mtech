from heapq import heappush, heappop
from collections import deque

## Import the project files
from Algorithms.ISearchAlgorithm import ISearchAlgorithm
from utils.GridEnvironment import GridEnvironment
from typing import List, Tuple

class GBFSearchAlgorithm(ISearchAlgorithm):
    """
    Greedy Best-First Search Algorithm implementation.

    This algorithm uses a priority queue to explore the search space based on a heuristic function.
    It expands the node with the lowest heuristic value, prioritizing the most promising paths towards the goal.

    Attributes:
        None

    Methods:
        search(grid_env: GridEnvironment) -> Tuple[List[Tuple[int, int]], int]:
            Performs the greedy best-first search on the given grid environment.

        heuristic(row: int, col: int) -> int:
            Calculates the heuristic value for a given cell in the grid.

    """
    def __init__(self, grid_env: GridEnvironment) -> None:
        super().__init__()

        self.grid_env = grid_env

    def search(self) -> Tuple[List[Tuple[int, int]], int]:
        """
        Performs the greedy best-first search on the given grid environment.

        Args:
            grid_env (GridEnvironment): The grid environment to search in.

        Returns:
            Tuple[List[Tuple[int, int]], int]: A tuple containing the path taken by the agent and the total path cost.

        """
        start = self.grid_env.start
        goal = self.grid_env.goal
        visited = set()
        pq = [(self.heuristic(*start), start)]
        came_from = {}
        cost_so_far = {start: 0}  # Store the cost of reaching each cell

        while pq:
            _, current = heappop(pq)
            if current == goal:
                # Reconstruct and print the path
                path = deque()
                total_cost = cost_so_far[current]
                while current != start:
                    path.appendleft(current)
                    current = came_from[current]
                path.appendleft(start)
                print("Path taken by the agent:", list(path))
                print("Total path cost:", total_cost)
                return list(path), total_cost
            visited.add(current)

            for next_cell in self.grid_env.get_adjacent_cells(*current, algorithm="greedy"):
                if next_cell in visited:  # Check if the cell has already been visited
                    continue  # Skip to the next iteration if the cell has been visited

                new_cost = cost_so_far[current] + 1  # Assuming each step has a cost of 1
                if next_cell not in cost_so_far or new_cost < cost_so_far[next_cell]:
                    cost_so_far[next_cell] = new_cost
                    print("Cost of next cell", cost_so_far[next_cell])
                    priority = new_cost + self.heuristic(*next_cell)
                    print("Priority of next cell", priority)
                    heappush(pq, (priority, next_cell))
                    came_from[next_cell] = current

    def heuristic(self, row: int, col: int) -> int:
        """
        Calculates the heuristic value for a given cell in the grid.

        The heuristic value is calculated based on the number of adjacent safe places, water bodies, and flooded roads.

        Args:
            row (int): The row index of the cell.
            col (int): The column index of the cell.

        Returns:
            int: The heuristic value for the cell.

        """
        score = 0
        for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            new_row, new_col = row + dr, col + dc
            if 0 <= new_row < self.grid_env.rows and 0 <= new_col < self.grid_env.cols:
                if self.grid_env.grid[new_row][new_col] == '.':
                    score += 5  # Add 5 points for adjacent safe places
                elif self.grid_env.grid[new_row][new_col] == '#':
                    score -= 5  # Deduct 5 points for adjacent water bodies
                elif self.grid_env.grid[new_row][new_col] == 'F':
                    score -= 3  # Deduct 3 points for flooded roads
        return score
