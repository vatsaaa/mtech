from heapq import heappush, heappop
from collections import deque
from typing import List, Tuple

## Import the project files
from utils.grid import track_time_and_space
from utils.GridEnvironment import GridEnvironment
from Algorithms.ISearchAlgorithm import ISearchAlgorithm


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

        time_complexity() -> str:
            Returns the time complexity of the algorithm.

        space_complexity() -> str:
            Returns the space complexity of the algorithm.

    """
    def __init__(self, grid_env: GridEnvironment) -> None:
        super().__init__()

        self.grid_env = grid_env
        self.tree = {}

    @track_time_and_space
    def search(self) -> Tuple[List[Tuple[int, int]], int, int]:
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
        total_nodes_expanded = 0
        total_branching_factor = 0
        depth_of_solution = 0

        while pq:
                print("Open List (Priority Queue):", pq)
                print("Closed List (Visited Set):", visited)
                priority, current = heappop(pq)  # Pop the item with lowest priority
                if current == goal:
                    # Reconstruct the path
                    path = deque()
                    total_cost = cost_so_far[current]
                    while current != start:
                        path.appendleft(current)
                        current = came_from[current]
                        depth_of_solution += 1  # Increment depth for each step towards the start node
                    path.appendleft(start)
                    break  # Exit the loop when the goal node is found
                    
                visited.add(current)
                total_nodes_expanded += 1

                successors = self.grid_env.get_adjacent_cells(*current, algorithm="greedy")
                total_branching_factor += len(successors)

                for next_cell in successors:
                    if next_cell in visited:  # Check if the cell has already been visited
                        continue  # Skip to the next iteration if the cell has been visited

                    new_cost = cost_so_far[current] + 1  # Assuming each step has a cost of 1
                    if next_cell not in cost_so_far or new_cost < cost_so_far[next_cell]:
                        cost_so_far[next_cell] = new_cost
                        # print("Cost of next cell", cost_so_far[next_cell])
                        priority = new_cost + self.heuristic(*next_cell)
                        print("Priority of next cell", priority)
                        heappush(pq, (priority, next_cell))  # Add the next cell to the priority queue
                        came_from[next_cell] = current

            # Store the depth of the optimal solution
        self.depth_of_solution = depth_of_solution
        print("Total Branching Factor:", round(total_branching_factor / total_nodes_expanded))
        print("Depth of the graph search tree is:", self.depth_of_solution)
            # Optionally, you can return the path and total cost if needed
        print("Space Complexity for GBFS Search is :", pq.count)
        return list(path), total_cost
    
 

    def heuristic(self, row: int, col: int) -> int:
        """
        Calculates heuristic value for a given cell in the grid. Considering
        if the adjacent cells are safe, water bodies, or flooded roads

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
