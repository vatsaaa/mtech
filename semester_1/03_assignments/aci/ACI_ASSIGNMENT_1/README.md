## How to run
cd semester_1/03_assignments/aci/ACI_ASSIGNMENT_1
```
python3 GENETIC_ALGORITHM/geneticalgo.py
```
OR

```
python3 GBFS/gbfs.py
```

-----GBFS PC----
function greedy_best_first_search(problem):
    Initialize an empty priority queue frontier
    Add the initial state of the problem to the frontier with priority f(initial_state) = h(initial_state)
    Initialize an empty set explored to keep track of explored states
    
    while frontier is not empty:
        current_node = pop the node with the highest priority from the frontier
        Add current_node.state to explored
        
        if current_node.state is the goal state:
            return current_node.path  # Return the path to the goal
        
        for each neighbor of current_node.state:
            if neighbor is not in explored and neighbor is not in frontier:
                Add neighbor to frontier with priority f(neighbor) = h(neighbor)
            else if neighbor is in frontier with a lower priority:
                Update neighbor's priority to f(neighbor) = h(neighbor)
                Update neighbor's parent to current_node
        
    return failure  # No solution found


initialize_population()
evaluate_fitness(population)
best_solution = find_best_solution(population)

while termination_criteria_not_met():
    next_generation = []

    while len(next_generation) < population_size:
        parent1 = select_parent(population)
        parent2 = select_parent(population)
        child = crossover(parent1, parent2)
        mutate(child)
        next_generation.append(child)

    population = next_generation
    evaluate_fitness(population)
    best_solution = find_best_solution(population)

return best_solution 


python ./main.py -b -d 

 python .\main.py -a -d
