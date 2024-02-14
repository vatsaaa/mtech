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
        # Uncomment the following lines if you have memory usage data available
        sns.scatterplot(x=self.gridSize, y=self.memoryConsumed)
        plt.title('Grid Size vs Memory Used')
        plt.xlabel('Grid Size')
        plt.ylabel('Memory Used')
        plt.show()

# # Example usage
# grid_size_list = [8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
# time_consumed_list = [1.2, 1.6, 1.7, 1.8, 2.0, 2.1, 2.3, 2.5, 2.6, 2.8]
# mem_consumed_list = [12, 16, 17, 18, 20, 21, 23, 25, 26, 28]

# # Create an instance of PlotPerformance
# plot_instance = PlotPerformance(gridSize=grid_size_list, timeConsumed=time_consumed_list, memoryConsumed=mem_consumed_list)

# # Call the plot method
# plot_instance.plot()







    # def plot(self):
    #     x_values = [data['grid_size'] for data in all_data]
    #     y_values = [data['execution_time'] for data in all_data]
 
    #     # Plot using Seaborn
    #     sns.scatterplot(x=x_values, y=y_values)
    #     plt.title('Grid Size vs Execution Time')
    #     plt.xlabel('Grid Size')
    #     plt.ylabel('Execution Time')
    #     plt.show()
 
    #     #----------plotting graph between gridsize vs memory consumed-------
 
    #     # Extract relevant fields for plotting
    #     x_values = [data['grid_size'] for data in all_data]
    #     y_values = [data['memory_usage'] for data in all_data]
 
    #     # Plot using Seaborn
    #     sns.scatterplot(x=x_values, y=y_values)
    #     plt.title('Grid Size vs Memory Used')
    #     plt.xlabel('Grid Size')
    #     plt.ylabel('Memory Used')
    #     plt.show()