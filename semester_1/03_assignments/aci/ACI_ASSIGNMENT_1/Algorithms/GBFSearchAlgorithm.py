from heapq import heappush, heappop
from collections import deque

## Import the project files
from Algorithms.ISearchAlgorithm import ISearchAlgorithm
from utils.GridEnvironment import GridEnvironment

class GBFSearchAlgorithm(ISearchAlgorithm):
    def search(self, grid_env: GridEnvironment): 
        start = grid_env.start
        goal = grid_env.goal
        visited = set()
        pq = [(grid_env.heuristic(*start), start)]
        came_from = {}
        cost_so_far = {start: 0}  # Store the cost of reaching each cell - revised by prasnejit

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

            for next_cell in grid_env.get_adjacent_cells(*current):
                if next_cell in visited:  # Check if the cell has already been visited
                    continue  # Skip to the next iteration if the cell has been visited

                new_cost = cost_so_far[current] + 1  # Assuming each step has a cost of 1
                if next_cell not in cost_so_far or new_cost < cost_so_far[next_cell]:
                    cost_so_far[next_cell] = new_cost
                    print("Cost of next cell", cost_so_far[next_cell])
                    priority = new_cost + grid_env.heuristic(*next_cell)
                    print("Priority of next cell", priority)
                    heappush(pq, (priority, next_cell))
                    came_from[next_cell] = current

    def heuristic(self, row, col):
        score = 0
        for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            new_row, new_col = row + dr, col + dc
            if 0 <= new_row < self.rows and 0 <= new_col < self.cols:
                if self.grid[new_row][new_col] == '.':
                    score += 5  # Add 5 points for adjacent safe places
                elif self.grid[new_row][new_col] == '#':
                    score -= 5  # Deduct 5 points for adjacent water bodies
                elif self.grid[new_row][new_col] == 'F':
                    score -= 3  # Deduct 3 points for flooded roads
        return score