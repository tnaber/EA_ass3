import numpy as np

def comparator(solution1, solution2):
    '''
    Comparator based on solutions rank and crowding distance
    '''

    if solution1.rank < solution2.rank:
        return 1
    if solution1.rank == solution2.rank and solution1.crowdingDistance > solution2.crowdingDistance:
        return 1

    return -1

def calculateHypervolume(solutions):
    '''
    Function for calculating hypervolume
    Input: list of Individuals assuming that these solutions do not dominate each other
    Output: single real value - the hypervolume formed by solutions with (1,1) as the reference point
    '''

    result = 0.0   
    sorted_solutions_fitnesses = [np.copy(s.fitness) for s in solutions]   
    bottom = 0
    left = 0

    #sort by first objective is descending order (from right to left)
    sorted_solutions_fitnesses = sorted(sorted_solutions_fitnesses, key = lambda x: x[0], reverse=True)
    for i, fitness in enumerate(sorted_solutions_fitnesses):
        current_hypervolume = (fitness[0] - left) * (fitness[1] - bottom)
        bottom = fitness[1]
        result += current_hypervolume

    return result

def round_to(n, precision):
    correction = 0.5 if n >= 0 else -0.5
    return round(int( n/precision+correction ) * precision, 3)