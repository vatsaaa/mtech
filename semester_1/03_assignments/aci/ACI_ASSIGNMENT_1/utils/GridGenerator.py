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
        def set_position(position, symbol):
            row, col = position
            if row < 0 or row > len(self.grid) or col < 0 or col > len(self.grid[0]):
                return None, "Please enter rows position between 0 to {rows} and columns between 0 to {cols}".format(rows=len(self.grid - 1), cols=len(self.grid[0] - 1))

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

        # Start position input and validation
        while True:
            start_input = input("\nEnter the start position (row,col): ")
            start_row, start_col = map(int, start_input.split(','))
            grid_with_start, error_message = set_position((start_row, start_col), 'S')
            if error_message:
                print(error_message)
            else:
                break

        # Goal position input and validation
        while True:
            goal_input = input("\nEnter the goal position (row,col): ")
            goal_row, goal_col = map(int, goal_input.split(','))
            grid_with_goal, error_message = set_position((goal_row, goal_col), 'G')
            if error_message:
                print(error_message)
            else:
                break

        # Print final grid with start and goal positions
        print("\nFinal grid with start and goal positions:")
        self.print_grid()

    def print_grid(self):
        for row in self.grid:
            print(' '.join(row))

