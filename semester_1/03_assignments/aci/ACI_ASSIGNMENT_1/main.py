import argparse
from pprint import pprint
from datetime import datetime

from utils.GridGenerator import GridGenerator
from utils.GridEnvironment import GridEnvironment
from utils.PersistPerformance import PersistPerformance
from Algorithms.GBFSearchAlgorithm import GBFSearchAlgorithm
from Algorithms.GeneticSearchAlgorithm import GeneticSearchAlgorithm
from Algorithms.SearchAlgorithmFactory import SearchAlgorithmFactory

def main():
    """
    This script runs a search algorithm on a grid environment.

    Usage:
        python main.py [-a | -b] [-d]

    Options:
        -a, --genetic   Run Genetic Algorithm
        -b, --gbfs      Run Greedy Best First Search
        -d, --display   Display grid

    Example:
        python main.py          ## Run both Genetic Algorithm and Greedy Best First Search
        python main.py -b -d    ## Run Greedy Best First Search
        python main.py -a       ## Run Genetic Algorithm
    """

    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-a", "--genetic", action="store_true", help="Run Genetic Algorithm")
    group.add_argument("-b", "--gbfs", action="store_true", help="Run Greedy Best First Search")
    parser.add_argument("-d", "--display", action="store_true", help="Display grid")
    args = parser.parse_args()

    dt_now = datetime.now()

    if(args.genetic == False) and (args.gbfs == False):
        print("Running both Genetic Algorithm and Greedy Best First Search")

        args.gbfs = True
        gbfs = SearchAlgorithmFactory.create_search_algorithm(args, None)
        gbfs_results, gbfs_etime, gbfs_emem = gbfs.search()
        gbfs_path = gbfs_results[0]
        gbfs_cost = gbfs_results[1]

        print("Path taken by the agent using Greedy Best First Search:", gbfs_path)
        print("Total path cost using Greedy Best First Search:", gbfs_cost)
        args.gbfs = False

        pp_gbfs = PersistPerformance(
            date_time=dt_now,
            execution_time=gbfs_etime,
            memory_usage=gbfs_emem,
            grid_shape=(gbfs.grid_env.rows, gbfs.grid_env.cols),
            start=gbfs.grid_env.start,
            goal=gbfs.grid_env.goal,
            algorithm="Greedy Best First Search"
        )
        # pp_gbfs.persist()
        pprint(pp_gbfs)

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
            grid_shape=(genetic_search.grid_env.rows, genetic_search.grid_env.cols),
            start=genetic_search.grid_env.start,
            goal=genetic_search.grid_env.goal,
            algorithm="Genetic Search"
        )
        # pp_gs.persist()
        pprint(pp_gs)


    else:
        search_algorithm = SearchAlgorithmFactory.create_search_algorithm(args, None)
        
        search_results, etime, emem = search_algorithm.search()
        path = search_results[0]
        cost = search_results[1]

        print("Path taken by the agent:", path)
        print("Total path cost:", cost)

        pp = PersistPerformance(
            date_time=dt_now,
            execution_time=etime,
            memory_usage=emem,
            grid_shape=(search_algorithm.grid_env.rows, search_algorithm.grid_env.cols),
            start=(0, 0),
            goal=(7, 7),
            algorithm=args.genetic if args.genetic else args.gbfs
        )
        pp.persist()

if __name__ == "__main__":
    main()
