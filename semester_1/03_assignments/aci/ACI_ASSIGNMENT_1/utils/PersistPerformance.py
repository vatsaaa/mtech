from pymongo import MongoClient

class PersistPerformance:
    def __init__(self, date_time, execution_time, memory_usage, grid_shape, start, goal):
        self.date_time = date_time
        self.execution_time = execution_time
        self.memory_usage = memory_usage
        self.grid_shape = grid_shape
        self.start = start
        self.goal = goal

    def persist(self):
        connection_string = "mongodb://<username>:<password>@<host>:<port>/<database>"

        # Create a MongoDB client
        client = MongoClient(connection_string)

        # Access the database
        db = client["your_database_name"]

        # Access the collection
        collection = db["your_collection_name"]

        # Create a document with performance data
        performance_data = {
            "date_time": self.date_time,
            "execution_time": self.execution_time,
            "memory_usage": self.memory_usage,
            "grid_shape": self.grid_shape,
            "start": self.start,
            "goal": self.goal
        }

        # Insert the document into the collection
        collection.insert_one(performance_data)

        # Close the MongoDB connection
        client.close()