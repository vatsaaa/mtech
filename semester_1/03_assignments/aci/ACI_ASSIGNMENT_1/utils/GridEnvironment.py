import matplotlib.pyplot as plt
import random
import sys

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

    def visualize(self, path):
         # Define colors for different elements
        colors = {'S': 'green', '.': 'white', '#': 'black', 'F': 'red', 'G': 'blue'}
        # Create a plot
        fig, ax = plt.subplots()

        # Plot each element with specified colors      
        for i in range(self.rows):
            for j in range(self.cols):
                ax.text(i, j, self.grid[i][j], ha='center', va='center', color=colors[self.grid[i][j]])

        # Customize ticks
        ax.set_xticks(range(len(self.grid[0])))
        ax.set_yticks(range(len(self.grid)))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        
        plt.title('S - Start, G - Goal')
        plt.xlabel('# - Water Bodies')
        plt.ylabel('F - Flooded Roads')

        # Add grid lines
        ax.grid(True, which='both', color='black', linewidth=1.5, linestyle='-', alpha=0.3)
        
        col = ['red', 'green', 'blue', 'cyan', 'magenta', 'orange', 'purple']
        colors = [col[random.randint(0, 6)] for _ in range(sys.getsizeof(path)-1)]
        X, Y = zip(*path)
        
        # Plot the path with different colors for each segment
        for i in range(sys.getsizeof(path)-1):
            plt.plot(X[i:i+2], Y[i:i+2], color=colors[i])
        
        # plt.plot(*zip(*path), marker='o', color='red', label='Path')
        plt.scatter(*path[0], marker='o', color='green', label='Start')
        plt.scatter(*path[-1], marker='o', color='blue', label='Goal')

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