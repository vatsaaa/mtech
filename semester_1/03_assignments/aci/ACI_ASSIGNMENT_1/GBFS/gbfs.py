## Import standard python libraries
from heapq import heappush, heappop
from collections import deque

## Import project files
from utils.GridEnvironment import GridEnvironment

## Greedy Best First Search Algorithm Impletation
def greedy_best_first_search(grid_env: GridEnvironment):
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
