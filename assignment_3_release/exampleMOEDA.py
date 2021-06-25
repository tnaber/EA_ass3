from src.MOEDA import MOEDA
from src import variation

L = 5 # number of (discrete) variables
populationSize = 100
EA = MOEDA(populationSize = populationSize, 
		numberOfVariables = L, 
		numberOfEvaluations = 10**4, 
		fitnessFunction = 'knapsack', 
		selection=variation.selection, variation_model=variation.marginalProductModel, mutation=variation.mutation, 
		tournamentSize = 2, mutationProb = 'auto',
		randomSeed = 30,
		elitism=True)
EA.evolve() # Run algorithm


#sizes of EA.hyperVolumeByGeneration and EA.numberOfEvaluationsByGeneration are equal
print('hypervolumes:', EA.hyperVolumeByGeneration) #print array of hypervolumes
print('#feval:', EA.numberOfEvaluationsByGeneration) #print array of #feval
EA.plotFronts(len(EA.numberOfEvaluationsByGeneration) - 1, "No_Elitism_5")