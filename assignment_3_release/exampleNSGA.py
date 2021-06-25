from src.NSGAII import NSGAII
from src import variation

L = 20 # number of (discrete) variables
populationSize = 100
EA = NSGAII(populationSize = populationSize, 
        numberOfVariables = L, 
        numberOfEvaluations = 10**4, 
        fitnessFunction = 'knapsack', 
        selection=variation.selection, crossover=variation.crossover, mutation=variation.mutation, 
        hillClimber = None,   
        tournamentSize = 2, crossoverProb=0.9, mutationProb = 'auto',
        randomSeed=42)
EA.evolve() # Run algorithm
print('hypervolumes:', EA.hyperVolumeByGeneration) #print array of hypervolumes
print('#feval:', EA.numberOfEvaluationsByGeneration) #print array of #feval
#sizes of EA.hyperVolumeByGeneration and EA.numberOfEvaluationsByGeneration are equal

