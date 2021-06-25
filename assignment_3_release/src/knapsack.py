import numpy as np

class KnapSack():
    def __init__(self, L, n_objectives):
        '''
        Reading knapsack isntances from file happens here
        '''

        filename = 'knapsack_instances/%d.txt' % (L)
        
        self.objects = []
        self.sum1, self.sum2 = 0, 0
        data = open(filename,'r').readlines()
        self.weight = int(data[0].split(' ')[1])
        self.r = []
        for line in data[1:]: #skip first line defining the graph structure
            line = line.strip().split(' ')
            line = [int(e) for e in line]
            self.objects.append((line))
            self.sum1 += line[1]
            self.sum2 += line[2]
            #objectives/weight ratio
            self.r.append((float(line[1])/float(line[0]), float(line[2])/float(line[0])))
        self.N = len(self.objects)

    def calculate(self, Individual):
        '''
        Method for calculating fitness given individual as input
        Input: Individual instance
        Output: tuple of objective values
        '''
        genotype, objectives = self.calculateFitnessForGenotype(Individual.genotype)
        Individual.fitness = tuple(objectives)
        Individual.genotype = np.copy(genotype)
        return objectives
    
    def calculateFitnessForGenotype(self, genotype):
        '''
        Method for calculating fitness given genotype as input
        Input: list or np.array of bits
        Output: tuple of objective values
        '''

        objectives = [0,0]
        weight = 0
        for i, object in enumerate(self.objects):
            if genotype[i] == 0:
                continue
            weight += object[0]
            objectives[0] += object[1]
            objectives[1] += object[2]
        
        #in case of constraint violation repair mechanism is needed
        while weight > self.weight:
            current_rs = [(k,self.r[k]) for k in range(self.N) if genotype[k] == 1]
            current_rs = [(r[0],np.max(r[1])) for r in current_rs]
            
            object_ro_remove = sorted(current_rs, key=lambda x:x[1])[0][0] #remove the object with the worst obj/weight ratio
            genotype[object_ro_remove] = 0 #remove object

            #adjust weight and objectives values
            weight -= self.objects[object_ro_remove][0]
            objectives[0] -= self.objects[object_ro_remove][1]
            objectives[1] -= self.objects[object_ro_remove][2]
        
        #values normalization
        objectives[0] /= float(self.sum1)
        objectives[1] /= float(self.sum2)       
            
        return genotype, objectives