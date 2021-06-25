import numpy as np
from tqdm import tqdm
from functools import cmp_to_key
import matplotlib
import matplotlib.pyplot as plt
from copy import copy

from .population import Population, dominates
from .utils import calculateHypervolume, comparator, round_to
from .knapsack import KnapSack
from .variation import sampleSolution


class MOEDA:
    def __init__(self, populationSize, numberOfVariables, numberOfEvaluations, fitnessFunction,
                 selection, variation_model, mutation,
                 hillClimber=None,
                 tournamentSize=2, mutationProb='auto',
                 randomSeed=42,
                 elitism=False):

        # Setting the random seed
        # np.random.seed(randomSeed)

        self.populationSize = populationSize
        self.numberOfVariables = numberOfVariables
        self.evaluationsBudget = numberOfEvaluations
        self.fitnessFunction = fitnessFunction
        self.elitism = elitism

        self.numberOfEvaluations = 0

        # initialize KnapSack for L variables with 2 objectives
        if fitnessFunction == 'knapsack':
            self.fitnessFunction = KnapSack(self.numberOfVariables, 2)

        self.population = Population(populationSize, numberOfVariables, self.fitnessFunction)
        self.numberOfEvaluations = populationSize

        self.offspringPopulation = Population(0, numberOfVariables, self.fitnessFunction)

        self.hillClimber = hillClimber
        self.selection = selection
        self.variation_model = variation_model
        self.mutation = mutation

        self.tournamentSize = tournamentSize
        if mutationProb == 'auto':
            self.mutationProb = 1.0 / numberOfVariables
        else:
            self.mutationProb = mutationProb

        self.fronts = []

        # Variables for monitoring evolution progress
        self.nonDominatedArchives = []  # list of nonDominatedArchives found in each generation
        self.hyperVolumeByGeneration = []  # list of hyperVolume values in each generation
        self.numberOfEvaluationsByGeneration = []  # list of #evaluations in each generation

        self.elitistArchive = set([])

    def updateElitistArchive(self, solution):
        '''
        Updating elitist archive with a new solution if necessary
        #####
        Input: a new solution (Individual class instance)
        Output: updated self.elitistArchive
        #####
        '''
        # print(round(solution.fitness[0], 2))
        iterable_archive = list(self.elitistArchive)
        dominated_elites = set([])


        for elite in iterable_archive:
            res = dominates(solution.fitness, elite.fitness)
            if res == 1: # Dominates
                dominated_elites.add(elite)
            elif res == -1: # Is dominated
                return

        self.elitistArchive = self.elitistArchive - dominated_elites
        self.elitistArchive.add(solution)

        return self.elitistArchive

    def crowdingDistanceAssignment(self, solutions):
        # initialize crowding distances to zero
        for solution in solutions:
            solution.crowdingDistance = 0

        for objectiveInd in range(2):  # 2 objectives

            objMax = np.max([s.fitness[objectiveInd] for s in solutions])  # max of current objective
            objMin = np.min([s.fitness[objectiveInd] for s in solutions])  # min of current objective
            objRange = objMax - objMin

            solutions = sorted(solutions, key=lambda x: x.fitness[objectiveInd])  # sort by current objective

            for i in range(1, len(solutions) - 1):
                if objRange == 0:  # special case when all objective values are equal
                    solutions[i].crowdingDistance = 0
                else:
                    solutions[i].crowdingDistance += (solutions[i + 1].fitness[objectiveInd] - solutions[i - 1].fitness[
                        objectiveInd]) / objRange

            # set crowding distance of extreme solutions to +inf
            solutions[0].crowdingDistance = solutions[-1].crowdingDistance = float("inf")

    def makeNewPopulation(self, population):

        # Optional Hill Climber for all solutions in the current population
        if self.hillClimber is not None:
            population, hillClimberEvals = self.hillClimber(population)
            self.numberOfEvaluations += hillClimberEvals

        # Learn distribution
        variation_model = self.variation_model(population)

        # Clear offspring of last generation
        self.offspringPopulation.solutions.clear()

        for i in range(0, len(self.population.solutions)):
            # Variation - sample offspring from estimated distribution
            self.offspringPopulation.solutions.append(sampleSolution(population, variation_model))

            # Mutation
            self.mutation(self.offspringPopulation.solutions[i], self.mutationProb)

            # Update fitness
            self.fitnessFunction.calculate(self.offspringPopulation.solutions[i])
            self.numberOfEvaluations += 1

    def evolve(self):
        evalsPerGeneration = self.populationSize
        if self.hillClimber is not None:
            evalsPerGeneration += self.populationSize * self.numberOfVariables

        for gen in tqdm(range(int(np.ceil(self.evaluationsBudget / float(evalsPerGeneration))))):

            # Check termination condition
            if self.numberOfEvaluations >= self.evaluationsBudget:
                break

            # add empty fronts list for the current generation
            self.fronts.append([[]])

            # Initial ranks and crowding distance assignment
            if gen == 0:
                self.crowdingDistanceAssignment(self.population.solutions)
                _ = self.population.fastNonDominatedSorting()

            # creates offsprings saved in self.offspringPopulation
            self.makeNewPopulation(self.population)

            # merge population and the newly obtained offspring population
            mergedPopulation = Population(0, self.numberOfVariables, self.fitnessFunction)
            mergedPopulation.merge(self.population, self.offspringPopulation)

            # save fronts of the merged population
            self.fronts[gen] = mergedPopulation.fastNonDominatedSorting()

            if self.elitism:
                # Updating the elitist archive with solutions from the non-dominated front
                for solution in self.fronts[gen][0]:
                    self.updateElitistArchive(solution)

                # calculate hypervolume of the elitist archive
                hypervolume = calculateHypervolume(self.elitistArchive)

                # # fill new population with solution from elitistArchive
                # self.population = Population(0, self.numberOfVariables, self.fitnessFunction)
                # for elite in self.elitistArchive:
                #     self.population.addSolution(elite)
            else:
                # calculate hypervolume of the non-dominated front of the current generation
                hypervolume = calculateHypervolume(self.fronts[gen][0])

            # fill new population with solutions from top fronts
            self.population = Population(0, self.numberOfVariables, self.fitnessFunction)
            for i, front in enumerate(self.fronts[gen]):
                if self.population.populationSize + len(front) > self.populationSize:
                    break

                self.crowdingDistanceAssignment(front)
                for f in front:
                    self.population.addSolution(f)

            # add missing solutions
            front = sorted(front, key=cmp_to_key(comparator), reverse=True)
            toAdd = self.populationSize - self.population.populationSize
            for i in range(toAdd):
                self.population.addSolution(front[i])

            # save hypervolume
            self.hyperVolumeByGeneration.append(hypervolume)
            self.numberOfEvaluationsByGeneration.append(self.numberOfEvaluations)




    def plotFronts(self, generation, title='fronts.png'):
        '''
        Function for plotting self.fronts[generation]
        '''

        fig, ax = plt.subplots(1, 1, figsize=(6, 6))

        plt.xlabel(r'$f_1$')
        plt.ylabel(r'$f_2$')
        plt.title('Generation %d' % generation)

        cmap = plt.cm.jet_r
        cmaplist = [cmap(i) for i in range(cmap.N)]
        N_colors = len(self.fronts[generation])
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list('Custom cmap', cmaplist, cmap.N)
        bounds = np.linspace(0, N_colors, N_colors + 1)
        norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

        x, y, tags = [], [], []
        for i, front in enumerate(self.fronts[generation]):
            obj1 = [solution.fitness[0] for solution in front]
            obj2 = [solution.fitness[1] for solution in front]
            x += obj1
            y += obj2
            tags += [i] * len(obj1)

        x, y, tags = np.array(x), np.array(y), np.array(tags)
        ind = np.where((x >= 0) & (y >= 0))[0]
        x, y, tags = x[ind], y[ind], tags[ind]
        plt.scatter(x, y, c=tags, cmap=cmap, norm=norm)
        plt.subplots_adjust(0.2, 0.2, 0.8, 0.8)

        ax = fig.add_axes([0.85, 0.1, 0.03, 0.8])
        cb = matplotlib.colorbar.ColorbarBase(ax, cmap=cmap, norm=norm,
                                              spacing='proportional', ticks=bounds + 0.5, boundaries=bounds,
                                              format='%1i')
        cb.ax.set_ylabel('Front index', size=12)
        plt.savefig('%s_%d.png' % (title, generation))
        plt.close()
