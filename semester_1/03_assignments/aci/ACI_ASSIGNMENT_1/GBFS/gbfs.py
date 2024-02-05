import heapq, pprint
from utils.grid import grid

class GridEnvironment:
    def __init__(self, grid):
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0])
        self.start = None
        self.goal = None
        self.find_start_and_goal()

    def find_start_and_goal(self):
        for i in range(self.rows):
            for j in range(self.cols):
                if self.grid[i][j] == 'S':
                    self.start = (i, j)
                    print(" Start position :", self.start)
                elif self.grid[i][j] == 'G':
                    self.goal = (i, j)
                    print(" Goal position :", self.goal)

    def is_valid_move(self, row, col):
        return (
            0 <= row < self.rows 
            and 0 <= col < self.cols 
            and self.grid[row][col] != '#' 
            and self.grid[row][col] != 'F'
        )

    def get_adjacent_cells(self, row, col):
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)] #degrees of freedom - need to revise by prasnejit
        adjacent_cells = []
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            if self.is_valid_move(new_row, new_col):
                print(" The move is valid for this %d and %d" % (new_row, new_col))
                adjacent_cells.append((new_row, new_col))
        return adjacent_cells

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

def greedy_best_first_search(grid_env):
    start = grid_env.start
    goal = grid_env.goal
    visited = set()
    pq = [(grid_env.heuristic(*start), start)]
    came_from = {}
    cost_so_far = {start: 0}  # Store the cost of reaching each cell - revised by prasnejit
    while pq:
        _, current = heapq.heappop(pq)
        if current == goal:
            # Reconstruct and print the path
            path = []
            total_cost = cost_so_far[current]
            while current != start:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            print("Path taken by the agent:", path)
            print("Total path cost:", total_cost)
            return path, total_cost
        visited.add(current)
        for next_cell in grid_env.get_adjacent_cells(*current):
            new_cost = cost_so_far[current] + 1  # Assuming each step has a cost of 1
            if next_cell not in cost_so_far or new_cost < cost_so_far[next_cell]:
                cost_so_far[next_cell] = new_cost
                print("Cost of next cell", cost_so_far[next_cell])
                priority = new_cost + grid_env.heuristic(*next_cell)
                print("Priority of next cell", priority)
                heapq.heappush(pq, (priority, next_cell))
                came_from[next_cell] = current

if __name__ == "__main__":
    env = GridEnvironment(grid)
    pprint.pprint(env.grid) # Print the grid
    
    path, total_cost = greedy_best_first_search(env)
    print("Path taken by the agent:",path)
    print("Total path cost:", total_cost)

