import matplotlib.pyplot as plt

class PlotPerformance:
    def __init__(self, gridSize: list, timeConsumed: list, memoryConsumed: list, algorithm: str):
        self.gridSize = gridSize
        self.timeConsumed = timeConsumed
        self.memoryConsumed = memoryConsumed
        self.algorithm = algorithm

    def plot_time(self):
        # Plotting line graph between grid size vs execution time
        plt.plot(self.gridSize, self.timeConsumed)
        plt.title('{algo} - Grid Size vs Execution Time'.format(algo=self.algorithm))
        plt.xlabel('Grid Size')
        plt.ylabel('Execution Time')
        plt.show()

    def plot_memory(self):
        # Plotting graph between grid size vs memory consumed
        plt.scatter(self.gridSize, self.memoryConsumed)
        plt.title('{algo} - Grid Size vs Memory Used'.format(algo=self.algorithm))
        plt.xlabel('Grid Size')
        plt.ylabel('Memory Used')
        plt.show()
