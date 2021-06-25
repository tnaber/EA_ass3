import numpy as np
from copy import copy

class Individual:
    def __init__(self, numberOfVariables, fitnessFunction):
        self.numberOfVariables = numberOfVariables
        self.genotype = np.random.randint(0, 2, numberOfVariables)
        fitnessFunction.calculate(self)

        #needed for non-dominated sorting
        self.rank = 0
        self.dominationCounter = 0
        self.dominatedSolutions = set([])

        #needed for crowding distance assignment
        self.crowdingDistance = 0

    def __str__(self):
        '''
        Function for printing individual calling print(solution)
        '''
        output = ''
        for bit in self.genotype:
              output += str(bit)
        output += ' | %.2f,%.2f | crowding distance = %.3f | rank=%d' % (self.fitness[0], self.fitness[1], self.crowdingDistance, self.rank)
        return output
    
    def __copy__(self):
        obj = type(self).__new__(self.__class__)
        obj.__dict__.update(self.__dict__)
        obj.genotype = np.copy(self.genotype)
        obj.dominatedSolutions = copy(self.dominatedSolutions)
        return obj