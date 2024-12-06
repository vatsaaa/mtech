import ssl as ssl
import matplotlib.pyplot as plt
from pymongo import cursor, MongoClient
import certifi
 
class PersistPerformance:
    def __init__(self, date_time: str, execution_time: str, memory_usage: str, total_nodes_expanded: int, grid_shape: str, start: str, goal: str, algorithm: str):
        self.date_time = date_time
        self.execution_time = execution_time
        self.memory_usage = memory_usage
        self.total_nodes_expanded = total_nodes_expanded
        self.grid_shape = grid_shape
        self.start = start
        self.goal = goal
        self.algorithm = algorithm
 
        # MongoDB connection parameters
        self.mongodb_uri = "mongodb+srv://saikrishnavedagiri567:7013227471@cluster-sai.8adbmzq.mongodb.net/"
        self.db = "AssignmentDB"
        self.collection = "TimeSpace"
  
    def persist(self) -> None:
        # Create a MongoDB client
        client = MongoClient(self.mongodb_uri, tlsCAFile=certifi.where())
 
        # Create a mongodb database object
        db = client[self.db]
 
        # Create the collection object
        collection = db[self.collection]
 
        # Create a document to be inserted
        document = {
            "date_time": self.date_time,
            "execution_time": self.execution_time,
            "memory_usage": self.memory_usage,
            "grid_size": self.grid_shape,
            "start": self.start,
            "goal": self.goal,
            "total_nodes_expanded": self.total_nodes_expanded,
            "algorithm": self.algorithm
        }
 
        # Insert the document into the collection
        collection.insert_one(document)
 
        # Close the MongoDB client
        client.close()

    def fetch_all(self) -> cursor.CursorType:
        client = MongoClient(self.mongodb_uri, tlsCAFile=certifi.where())

        # Connect to the database
        db = client[self.db]
 
        # Connect to the collection
        collection = db[self.collection]
 
        # Fetch all documents from the collection
        documents = collection.find()

        return documents

    def fetch_by(self, by: str, value: str) -> cursor.CursorType:
        client = MongoClient(self.mongodb_uri, tlsCAFile=certifi.where())

        # Connect to the database
        db = client[self.db]
 
        # Connect to the collection
        collection = db[self.collection]
 
        # Fetch documents from the collection based on the specified criteria
        documents = collection.find({by: value}).sort("date_time", -1)

        return documents

    def __repr__(self):
        return f"Date: {self.date_time}, Execution Time: {self.execution_time}, Memory Usage: {self.memory_usage}, Grid Size: {self.grid_shape}, Start: {self.start}, Goal: {self.goal}, Algorithm: {self.algorithm}"
    
    def fetch_by(self, by: str, value: str) -> cursor.CursorType:
        # Create a MongoDB client
        client = MongoClient(self.mongodb_uri, tlsCAFile=certifi.where())

        # Connect to the database
        db = client[self.db]
 
        # Connect to the collection
        collection = db[self.collection]
 
        # Fetch all documents from the collection in order of date_time
        documents = collection.find({by: value}).sort("date_time", -1)

        return documents

    def __repr__(self):
        return f"Date: {self.date_time}, Execution Time: {self.execution_time}, Memory Usage: {self.memory_usage}, Grid Size: {self.grid_shape}, Start: {self.start}, Goal: {self.goal}, Algorithm: {self.algorithm}"