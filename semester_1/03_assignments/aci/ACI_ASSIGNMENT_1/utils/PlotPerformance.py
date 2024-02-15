import matplotlib.pyplot as plt
import seaborn as sns

class PlotPerformance:
    def __init__(self, gridSize: list, timeConsumed: list, memoryConsumed: list):
        self.gridSize = gridSize
        self.timeConsumed = timeConsumed
        self.memoryConsumed = memoryConsumed

    def plot(self):
        # Plotting graph between grid size vs execution time
        sns.scatterplot(x=self.gridSize, y=self.timeConsumed)
        plt.title('Grid Size vs Execution Time')
        plt.xlabel('Grid Size')
        plt.ylabel('Execution Time')
        plt.show()

        # ----------plotting graph between grid size vs memory consumed-------
        sns.scatterplot(x=self.gridSize, y=self.memoryConsumed)
        plt.title('Grid Size vs Memory Used')
        plt.xlabel('Grid Size')
        plt.ylabel('Memory Used')
        plt.show()
