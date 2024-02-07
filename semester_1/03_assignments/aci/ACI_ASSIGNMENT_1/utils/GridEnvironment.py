class GridEnvironment:
    def __init__(self, grid: list[list[str]], display: bool = False):
        def find_start_and_goal():
            for i in range(self.rows):
                for j in range(self.cols):
                    if self.grid[i][j] == 'S':
                        self.start = tuple((i, j))
                    elif self.grid[i][j] == 'G':
                        self.goal = tuple((i, j))
            print("Start {start}, Goal {goal}:".format(start=self.start, goal=self.goal))

        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0])
        self.display = display
        find_start_and_goal()

    def is_valid_move(self, row, col):
        return (
            0 <= row < self.rows 
            and 0 <= col < self.cols 
            and self.grid[row][col] != '#' 
            and self.grid[row][col] != 'F'
        )

    def get_adjacent_cells(self, row, col, algorithm=None):
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)] #degrees of freedom - need to revise by prasnejit
        adjacent_cells = [] # Return empty list of adjacent cells if all moves are invalid
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            if self.is_valid_move(new_row, new_col):
                if algorithm == "greedy" and self.display:
                    print(" The move is valid for this %d and %d" % (new_row, new_col))
                adjacent_cells.append(tuple((new_row, new_col)))
        return adjacent_cells
    
    def goal_reached(self, row, col):
        return (row, col) == self.goal