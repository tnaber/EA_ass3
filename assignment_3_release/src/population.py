from copy import copy

from .individual import Individual

def dominates(fitness_A, fitness_B):
    '''
    Returns 1 if A dominates B, -1 if B dominates A, 0 otherwise
    '''
    
    if fitness_A == fitness_B:
        return 0
    
    if fitness_A[0] >= fitness_B[0] and fitness_A[1] >= fitness_B[1]:
        return 1

    if fitness_A[0] <= fitness_B[0] and fitness_A[1] <= fitness_B[1]:
        return -1
    
    return 0

class Population:

    def __init__(self, populationSize, numberOfVariables, fitnessFunction):
        self.populationSize = populationSize
        self.numberOfVariables = numberOfVariables
        self.fitnessFunction = fitnessFunction
        self.solutions = [Individual(numberOfVariables, fitnessFunction) for i in range(populationSize)]
        
    def merge(self, population_A, population_B):
        '''
        Make new population as merge of two given populations
        '''
        self.populationSize = population_A.populationSize + population_B.populationSize
        self.numberOfVariables = population_A.numberOfVariables
        self.fitnessFunction = population_A.fitnessFunction
        self.solutions = population_A.solutions + population_B.solutions

    def addSolution(self, solution):
        '''
        Adding new solution to population
        '''
        self.solutions.append(copy(solution))
        self.populationSize += 1
    
    def fastNonDominatedSorting(self):
        '''
        Fast non-dominating sorting algorithm
        It sorts self.solutions into non dominated fronts
        #####
        Input: empty. Fast non-dominated sorting is applied to self.solutions
        Output: list of lists (variable "fronts")
        fronts[0] is the best front, fronts[-1] is the worst one
        #####
        '''

        fronts = [[]]
        
        ################
        ##Your code here



        ################
        
        for p in self.solutions:
            p.dominatedSolutions = []
            p.dominationCounter = 0
      
            for q in self.solutions:
                if dominates(p.fitness, q.fitness) == 1:
                    p.dominatedSolutions.append(q)
                elif dominates(p.fitness, q.fitness) == -1:
                    p.dominationCounter += 1
      
            if p.dominationCounter == 0:
                p.rank = 1
                fronts[0].append(p)
      
        frontInd = 0
        while True:
            nextFront = []
            for p in fronts[frontInd]:
                for q in p.dominatedSolutions:
                    q.dominationCounter -= 1
                    if q.dominationCounter == 0:
                        q.rank = frontInd + 1                       
                        nextFront.append(q)
                        
            if len(nextFront) == 0:
                break

            fronts.append(nextFront)
            frontInd += 1
            
        return fronts