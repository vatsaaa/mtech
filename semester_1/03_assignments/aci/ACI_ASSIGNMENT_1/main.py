import argparse
from pprint import pprint
from datetime import datetime

from utils.GridGenerator import GridGenerator
from utils.GridEnvironment import GridEnvironment
from utils.PersistPerformance import PersistPerformance
from utils.PlotPerformance import PlotPerformance
from Algorithms.GBFSearchAlgorithm import GBFSearchAlgorithm
from Algorithms.GeneticSearchAlgorithm import GeneticSearchAlgorithm
from Algorithms.SearchAlgorithmFactory import SearchAlgorithmFactory

def fetch_gbfs_records():
    dummy_pp = PersistPerformance(
        date_time='',
        execution_time='',
        memory_usage='',
        grid_shape=(0, 0),
        start=(0, 0),
        goal=(0, 0),
        total_nodes_expanded=0,
        algorithm=''
    )
    gbfs_records = dummy_pp.fetch_by("algorithm", "Greedy Best First Search")
    return gbfs_records

def fetch_gs_records():
    dummy_pp = PersistPerformance(
        date_time='',
        execution_time='',
        memory_usage='',
        grid_shape=(0, 0),
        start=(0, 0),
        goal=(0, 0),
        total_nodes_expanded=0,
        algorithm=''
    )
    gs_records = dummy_pp.fetch_by("algorithm", "Genetic Search")
    return gs_records

def main():
    """
    This script runs a search algorithm on a grid environment.

    Usage:
        python main.py [-a | -b] [-d] [-p]

    Options:
        -a, --genetic   Run Genetic Algorithm
        -b, --gbfs      Run Greedy Best First Search
        -d, --display   Display grid
        -p, --plot      Plot performance graphs

    Example:
        python main.py          ## Run both Genetic Algorithm and Greedy Best First Search
        python main.py -b -d    ## Run Greedy Best First Search
        python main.py -a       ## Run Genetic Algorithm
        python main.py -p       ## Plot performance graphs
    """

    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-a", "--genetic", action="store_true", help="Run Genetic Algorithm")
    group.add_argument("-b", "--gbfs", action="store_true", help="Run Greedy Best First Search")
    parser.add_argument("-d", "--display", action="store_true", help="Display grid")
    parser.add_argument("-p", "--plot", action="store_true", help="Plot performance graphs")
    
    args = parser.parse_args()
    
    dt_now = datetime.now()

    if(args.genetic == False) and (args.gbfs == False) and (args.plot == False):
        print("Running both Genetic Algorithm and Greedy Best First Search")

        args.gbfs = True
        gbfs = SearchAlgorithmFactory.create_search_algorithm(args, None)
        gbfs_results, gbfs_etime, gbfs_emem = gbfs.search()
        gbfs_path = gbfs_results[0]
        gbfs_cost = gbfs_results[1]

        print("Path taken by the agent using Greedy Best First Search:", gbfs_path)
        print("Total path cost using Greedy Best First Search:", gbfs_cost)
        print("Total Memory Usage Greedy Best First Search:", gbfs.total_nodes_expanded)
        args.gbfs = False

        pp_gbfs = PersistPerformance(
            date_time=dt_now,
            execution_time=gbfs_etime,
            memory_usage=gbfs_emem,
            total_nodes_expanded=gbfs.total_nodes_expanded,
            grid_shape=(gbfs.grid_env.rows, gbfs.grid_env.cols),
            start=gbfs.grid_env.start,
            goal=gbfs.grid_env.goal,
            algorithm="Greedy Best First Search"
        )
        pp_gbfs.persist()

        args.genetic = True
        genetic_search = SearchAlgorithmFactory.create_search_algorithm(args, gbfs.grid_env.grid)
        gs_results, gs_etime, gs_emem = genetic_search.search()
        gs_path = gs_results[0]
        gs_cost = gs_results[1]

        print("Path taken by the agent using Genetic Algorithm:", gs_path)
        print("Total path cost using Genetic Algorithm:", gs_cost)

        pp_gs = PersistPerformance(
            date_time=dt_now,
            execution_time=gs_etime,
            memory_usage=gs_emem,
            total_nodes_expanded=genetic_search.total_nodes_expanded,
            grid_shape=(genetic_search.grid_env.rows, genetic_search.grid_env.cols),
            start=genetic_search.grid_env.start,
            goal=genetic_search.grid_env.goal,
            algorithm="Genetic Search"
        )
        pp_gs.persist()
    elif args.gbfs or args.genetic:
        search_algorithm = SearchAlgorithmFactory.create_search_algorithm(args, None)
        
        search_results, etime, emem = search_algorithm.search()
        path = search_results[0]
        cost = search_results[1]

        print("Path taken by the agent:", path)
        print("Total path cost:", cost)
        print("Total memory :", search_algorithm.total_nodes_expanded)
        algorithm="Genetic Search" if args.genetic else "Greedy Best First Search"
        pp = PersistPerformance(
            date_time=dt_now,
            execution_time=etime,
            memory_usage=emem,
            total_nodes_expanded=search_algorithm.total_nodes_expanded,
            grid_shape=(search_algorithm.grid_env.rows, search_algorithm.grid_env.cols),
            start=(0, 0),
            goal=(7, 7),
            algorithm=algorithm
        )
        pp.persist()
    elif args.plot:
        gbfs_grid_size_list = []
        gbfs_exe_time_list = []
        gbfs_mem_con_list = []
        count_gbfs = 0
        for data in fetch_gbfs_records():
            count_gbfs += 1
            for key, value in data.items():
                if key == "grid_size":
                    gbfs_grid_size_list.append(value[0])
                elif key == "execution_time":
                    gbfs_exe_time_list.append(value)
                elif key == "memory_usage":
                    gbfs_mem_con_list.append(value)
        if args.display:
            print("Number of Greedy Best First Search records:", count_gbfs)

        gs_grid_size_list = []
        gs_exe_time_list = []
        gs_mem_con_list = []
        count_gs = 0
        for data in fetch_gs_records():
            count_gs += 1
            for key, value in data.items():
                if key == "grid_size":
                    gs_grid_size_list.append(value[0])
                elif key == "execution_time":
                    gs_exe_time_list.append(value)
                elif key == "memory_usage":
                    gs_mem_con_list.append(value)
        if args.display:
            ("Number of Genetic Search records:", count_gs)

        # Create an instance of PlotPerformance
        plot_instance = PlotPerformance(gridSize=gbfs_grid_size_list, timeConsumed=gbfs_exe_time_list, memoryConsumed=gbfs_mem_con_list, algorithm="Greedy Best First Search")
        plot_instance.plot_time()
        plot_instance.plot_memory()
    else:
        print("Invalid arguments. Please refer to the help section below.")
        parser.print_help()

if __name__ == "__main__":
    main()
