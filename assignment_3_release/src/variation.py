import numpy as np
import scipy
from functools import cmp_to_key
from copy import copy
from collections import Counter
from scipy.stats import entropy

from .individual import Individual
from .population import Population
from .utils import comparator

def crossover(parent1, parent2):
    '''
    Uniform crossover
    '''
    positions = np.random.randint(0, 2, parent1.numberOfVariables)
    positions = np.where(positions == 0)[0]
    tmp1 = np.copy(parent1.genotype)
    parent1.genotype[positions] = parent2.genotype[positions]
    parent2.genotype[positions] = tmp1[positions]
    
def mutation(solution, prob):
    '''
    Simple bit-wise mutation (bits flipped with probability=prob)
    '''
    for i in range(solution.numberOfVariables):
        if np.random.uniform(0,1) < prob:
            solution.genotype[i] = 1 - solution.genotype[i]
    
def selection(population, selectionSize, tournamentSize):
    '''
    Tournament selection
    '''

    selected = Population(0, population.numberOfVariables, population.fitnessFunction)
    for i in range(selectionSize):
        contestantsIndices = np.random.choice(population.populationSize, tournamentSize, replace=False) #select tournamentSize solutions without replacement
        tournament = [population.solutions[ind] for ind in contestantsIndices]
        tournament = sorted(tournament, key = cmp_to_key(comparator), reverse = True)
        winner = tournament[0]
        selected.solutions.append(copy(winner))
        selected.populationSize += 1
    return selected

def oneIterHillClimber(population):
    '''
    one iteration of random direction hill climber
    '''
    hillClimbed = Population(0, population.numberOfVariables, population.fitnessFunction)
    evals = 0
    for i in range(population.populationSize):
        hillClimbedSolution = copy(population.solutions[i])        
        alpha = np.random.uniform(0, 1)
        cur_score = alpha*hillClimbedSolution.fitness[0] + (1-alpha)*hillClimbedSolution.fitness[1]
        #print('initial fitness', hillClimbedSolution.fitness)
        for j in range(population.numberOfVariables):
            for value in [0, 1]:
                if hillClimbedSolution.genotype[j] != value:
                    backup_value = hillClimbedSolution.genotype[j]
                    hillClimbedSolution.genotype[j] = value
                    new_objectives = population.fitnessFunction.calculate(hillClimbedSolution)
                    new_score = alpha*new_objectives[0] + (1-alpha)*new_objectives[1]
                    #print(cur_score, new_score)
                    if new_score > cur_score:
                        hillClimbedSolution.fitness = new_objectives
                        cur_score = new_score
                    else:
                        hillClimbedSolution.genotype[j] = backup_value
                    evals += 1
        #print('final fitness', hillClimbedSolution.fitness)
        hillClimbed.solutions.append(copy(hillClimbedSolution))
        hillClimbed.populationSize += 1
    return hillClimbed, evals

def marginalProductModel(population):
	# Initialize marginal product model (MPM) as a list of all univariate marginals
	# Stored as a list of pairs where each pair describes a marginal and its description length (DL)
	mpm = [([i],getDescriptionLength(population,[i])) for i in range(population.numberOfVariables)]
	# Merge marginals until MDL can no longer be improved
	while( True ):
		np.random.shuffle(mpm)
		mdl = np.inf
		# Attempt each possible merge
		for i in range(len(mpm)):
			x,dlx = mpm[i]
			for j in range(i):
				y,dly = mpm[j]
				# Compute DL of merging marginals x and y
				dlxy = getDescriptionLength(population,x+y)
				# Merge improves MDL and is best merge found so far
				if( dlxy < dlx+dly and dlxy-dlx-dly < mdl ):
					mdl = dlxy-dlx-dly
					best_merge = (i,j,dlxy)
		# No improvement of MDL was found
		if mdl == np.inf:
			break
		# Improvement of MDL was found
		else:
			i,j,dl = best_merge
			x,_ = mpm[i]
			y,_ = mpm[j]
			for s in sorted([i,j], reverse = True):
				del mpm[s]
			# Insert merged set into MPM
			mpm.append((x+y,dl))
	#print([x[0] for x in mpm])
	return mpm

def getDescriptionLength(population,var_indices):
	# Create list of population (as bitstrings) restricted to problem indices in 'var_indices'
	population_str = [tuple([individual.genotype[i] for i in var_indices]) for individual in population.solutions]
	# Count frequencies of patterns in population, then compute entropy
	frequencies = Counter(population_str)
	entropy = scipy.stats.entropy([frequencies[pattern]/population.populationSize for pattern in frequencies],base=2)
	compressed_population_complexity = entropy * population.populationSize
	model_complexity = np.log2(population.populationSize+1)	* (2**len(var_indices)-1)
	return model_complexity + compressed_population_complexity

def sampleSolution(population,mpm):
	'''
	Samples a new solution by using marginals from the MPM as mask and the population as donors
	The MPM and population together encode the discrete distribution estimated by the EDA
	'''
	# Copy an arbitrary solution; all genes will be overwritten
	offspring = copy(population.solutions[0])
	for marginal in mpm:
		variable_indices,_ = marginal
		# Select random donor from the population
		donor_index = np.random.randint(population.populationSize)
		# Use marginal as mask to copy genotype
		for x in variable_indices:
			offspring.genotype[x] = population.solutions[donor_index].genotype[x]
	return offspring

