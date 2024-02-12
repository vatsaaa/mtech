import matplotlib.pyplot as plt
from pymongo import MongoClient
import seaborn as sns
 
class PersistPerformance:
    def __init__(self, date_time: str, execution_time: str, memory_usage: str, grid_shape: str, start: str, goal: str, algorithm: str):
        self.date_time = date_time
        self.execution_time = execution_time
        self.memory_usage = memory_usage
        self.grid_shape = grid_shape
        self.start = start
        self.goal = goal
        self.algorithm = algorithm
 
        # MongoDB connection parameters
        self.mongodb_uri = "mongodb+srv://saikrishnavedagiri567:7013227471@cluster-sai.8adbmzq.mongodb.net/"
        self.db = "AssignmentDB"
        self.collection = "TimeSpace"
  
    def persist(self):
        # Create a MongoDB client
        client = MongoClient(self.mongodb_uri)
 
        # Connect to the database
        db = client[self.db]
 
        # Connect to the collection
        collection = db[self.collection]
 
        # Create a document to be inserted
        document = {
            "date_time": self.date_time,
            "execution_time": self.execution_time,
            "memory_usage": self.memory_usage,
            "grid_size": self.grid_shape,
            "start": self.start,
            "goal": self.goal,
            "algorithm": self.algorithm
        }
 
        # Insert the document into the collection
        collection.insert_one(document)
 
        # Close the MongoDB client
        client.close()

    def fetch(self):
        # Create a MongoDB client
        client = MongoClient(self.mongodb_uri)
 
        # Connect to the database
        db = client[self.db]
 
        # Connect to the collection
        collection = db[self.collection]
 
        # Fetch all documents from the collection
        documents = collection.find()
 
        client.close()

        return documents
    
    def __repr__(self):
        return f"Date: {self.date_time}, Execution Time: {self.execution_time}, Memory Usage: {self.memory_usage}, Grid Size: {self.grid_shape}, Start: {self.start}, Goal: {self.goal}, Algorithm: {self.algorithm}"

"""
    def plot(self):
        x_values = [data['grid_size'] for data in all_data]
        y_values = [data['execution_time'] for data in all_data]
 
        # Plot using Seaborn
        sns.scatterplot(x=x_values, y=y_values)
        plt.title('Grid Size vs Execution Time')
        plt.xlabel('Grid Size')
        plt.ylabel('Execution Time')
        plt.show()
 
        #----------plotting graph between gridsize vs memory consumed-------
 
        # Extract relevant fields for plotting
        x_values = [data['grid_size'] for data in all_data]
        y_values = [data['memory_usage'] for data in all_data]
 
        # Plot using Seaborn
        sns.scatterplot(x=x_values, y=y_values)
        plt.title('Grid Size vs Memory Used')
        plt.xlabel('Grid Size')
        plt.ylabel('Memory Used')
        plt.show()
 """