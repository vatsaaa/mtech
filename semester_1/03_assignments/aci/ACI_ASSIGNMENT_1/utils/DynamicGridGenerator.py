#DynamicGrid Generator 
import random

def dynamic_grid_generator(rows, cols):
    total_cells = rows * cols

    # Ensure at least 1 'F' obstacle and 1 '#' obstacle
    min_f_obstacles = 1
    min_hash_obstacles = 1

    # Calculate the maximum number of obstacles based on the total number of cells
    max_f_obstacles = int(total_cells / 3)  # Maximum 1/3 of cells for 'F' obstacles
    max_hash_obstacles = int(total_cells / 3)  # Maximum 1/3 of cells for '#' obstacles

    # Randomly choose the number of 'F' and '#' obstacles within the specified range
    num_f_obstacles = random.randint(min_f_obstacles, max_f_obstacles)
    num_hash_obstacles = random.randint(min_hash_obstacles, max_hash_obstacles)

    # Initialize grid with all safe places ('.')
    grid = [['.' for _ in range(cols)] for _ in range(rows)]

    # Place 'F' obstacles randomly
    for _ in range(num_f_obstacles):
        row, col = random.randint(0, rows - 1), random.randint(0, cols - 1)
        grid[row][col] = 'F'

    # Place '#' obstacles randomly
    for _ in range(num_hash_obstacles):
        row, col = random.randint(0, rows - 1), random.randint(0, cols - 1)
        # Ensure the cell is not already occupied and not occupied by 'F' obstacle
        while grid[row][col] != '.':
            row, col = random.randint(0, rows - 1), random.randint(0, cols - 1)
        grid[row][col] = '#'

    # Place 'S' (start) position
    start_row, start_col = random.randint(0, rows - 1), random.randint(0, cols - 1)
    grid[start_row][start_col] = 'S'

    # Place 'G' (goal) position
    goal_row, goal_col = random.randint(0, rows - 1), random.randint(0, cols - 1)
    while (goal_row, goal_col) == (start_row, start_col):
        goal_row, goal_col = random.randint(0, rows - 1), random.randint(0, cols - 1)
    grid[goal_row][goal_col] = 'G'

    return grid

# Generate random number of rows and columns (between 1 and 18)
rows = cols = random.randint(1, 18)
grid = dynamic_grid_generator(rows, cols)

# Print the generated grid
for row in grid:
    print(' '.join(row))
