# generateGrid.py

class GridGenerator:
    def __init__(self):
        self.grid = [
            ['.', '.', '.', '.', '#', '#', '.', '.'],
            ['.', 'F', 'F', '.', '.', '.', '.', 'F'],
            ['.', '.', '.', '.', '.', '.', '.', '.'],
            ['#', '#', '.', 'F', '.', '.', '.', '.'],
            ['.', 'F', '.', '.', '.', '.', '#', '#'],
            ['.', '.', '.', 'F', '.', '.', '.', '.'],
            ['.', '#', '#', '.', '.', 'F', '.', '.'],
            ['.', '.', '.', '.', '.', 'F', '.', '.']
        ]

    def generate_grid(self):
        # Start position input and validation
        while True:
            start_input = input("\nEnter the start position (row,col): ")
            start_row, start_col = map(int, start_input.split(','))
            grid_with_start, error_message = self.set_position((start_row, start_col), 'S')
            if error_message:
                print(error_message)
            else:
                break

        # Goal position input and validation
        while True:
            goal_input = input("\nEnter the goal position (row,col): ")
            goal_row, goal_col = map(int, goal_input.split(','))
            grid_with_goal, error_message = self.set_position((goal_row, goal_col), 'G')
            if error_message:
                print(error_message)
            else:
                break

        # Print final grid with start and goal positions
        print("\nFinal grid with start and goal positions:")
        self.print_grid()

    def set_position(self, position, symbol):
        row, col = position
        if row < 0 or row > 7 or col < 0 or col > 7:
            return None, f"Invalid input! Row and column indices must be between 0 and 7."
        if self.grid[row][col] == '#':
            self.print_grid()
            return None, f"Invalid {symbol} position! This cell is occupied by a water body ('#')."
        if self.grid[row][col] == 'F':
            self.print_grid()
            return None, f"Invalid {symbol} position! This cell is flooded ('F')."
        if self.grid[row][col] == 'S':
            self.print_grid()
            return None, f"Invalid {symbol} position! The start and the Goal position cannot be same."
        self.grid[row][col] = symbol
        return self.grid, None

    def print_grid(self):
        for row in self.grid:
            print(' '.join(row))

