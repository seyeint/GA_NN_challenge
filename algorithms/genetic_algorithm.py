import logging
import numpy as np
import utils as uls
from functools import reduce
import matplotlib.pyplot as plt
import scipy

from random_search import RandomSearch
from solutions.solution import Solution


class GeneticAlgorithm(RandomSearch):
    def __init__(self, problem_instance, random_state, population_size,
                 selection, crossover, p_c, mutation, p_m):
        RandomSearch.__init__(self, problem_instance, random_state)
        self.population_size = population_size
        self.selection = selection
        self.crossover = crossover
        self.p_c = p_c
        self.mutation = mutation
        self.p_m = p_m

    def initialize(self):
        self.population = self._generate_random_valid_solutions()
        self.best_solution = self._get_elite(self.population)

    def search(self, n_iterations, report=False, log=False):
        fitnesses=[]
        if log:
            log_event = [self.problem_instance.__class__, id(self._random_state), __name__]
            logger = logging.getLogger(','.join(list(map(str, log_event))))

        elite = self.best_solution
        n = len(self.population) - 4 #we choose 4 because is close to 10%

        for iteration in range(n_iterations):
            offsprings = []

            copy = np.array([parent.fitness for parent in self.population])
            ind_max1 = np.argmax(copy)
            copy[ind_max1] = 0
            ind_max2 = np.argmax(copy)
            copy[ind_max2] = 0
            ind_max3 = np.argmax(copy)
            copy[ind_max3] = 0
            ind_max4 = np.argmax(copy)

            top = np.array([ind_max1, ind_max2, ind_max3, ind_max4])  # index
            top_people = np.array([self.population[ind_max1], self.population[ind_max2], self.population[ind_max3],
                                    self.population[ind_max4]])  # real pop
            countx = 0
            np.delete(self.population, top)
            print(len(self.population))
            if iteration==95: #this was just a final try to get something after the shift
                for i in self.population:
                    if i not in top_people:
                        if self._random_state.uniform()<0.5:
                            i=np.random.choice(top_people, 1, p=[0.4, 0.3, 0.2, 0.1])[0]
                            countx+=1

            print('COUNTX',countx)


            while len(offsprings) < n:

                off1, off2 = p1, p2 = [
                    self.selection(self.population, self.problem_instance.minimization, self._random_state) for _ in range(2)]
                if self._random_state.uniform() < self.p_c:
                    off1, off2 = self._crossover(p1, p2)
                if self._random_state.uniform() < self.p_m:
                    off2 = self._mutation(off2)
                    off1 = self._mutation(off1)

                if not (hasattr(off1, 'fitness') and hasattr(off2, 'fitness')):
                    self.problem_instance.evaluate(off1)
                    self.problem_instance.evaluate(off2)

                offsprings.extend([off1, off2])

            self.population = np.append(self.population, top_people)
            print(len(self.population))

            while len(offsprings) > len(self.population):
                offsprings.pop()

            elite_offspring = self._get_elite(offsprings)
            elite = self._get_best(elite, elite_offspring)

            #Crowding try
            #rad = 30
            #for j in range(len(offsprings) - 1):
             #   for k in range(1, len(offsprings)):
              #      distance = np.linalg.norm(offsprings[j].representation - offsprings[k].representation)
               #     print(distance)

                #    if distance < rad:
                 #       if distance > 0:  # to not delete when comparing to himself
                  #          if offsprings[k].fitness < offsprings[j].fitness:
                   #             np.delete(offsprings, k)
                    #            count += 1
            #print('Melted:', count)
            #print(len(offsprings))
            ########################################


            #trying genotypic distance instead of phenotipic to share fitness
            #rad=70
            #count = 0
            #for j in range(len(offsprings) - 1):
             #   for k in range(1, len(offsprings)):
              #      distance = np.linalg.norm(offsprings[j].representation - offsprings[k].representation) #wannabe genotypic
               #     if distance < rad:
                #        if distance > 0: #to not work when comparing to himself
                 #           if offsprings[k].fitness < offsprings[j].fitness:
                  #              offsprings[j] = offsprings[k]
                   #             count += 1
            #print('DownGraded :', count)
            ############################################

            #Here we would have low p_m in main, and when we see a phenotypic shift we would act!
            #if self._phenotypic_diversity_shift(offsprings)<0:
               # self.p_m = 0.99
                #self.p_c=0.7
                #print('MUTATION')



            #Phenotypic distance and fitness sharing
            niche_radius = 0.001

            if iteration < n_iterations-1:#to not touch in the fitness of the last gen
                for j in range(len(offsprings) - 1):
                    niche_count = 0
                    for k in range(len(offsprings)-1):
                        distance = abs(offsprings[j].fitness - offsprings[k].fitness)
                        if distance < niche_radius and distance!=0:
                            share_function = 1-((distance/niche_radius)**2)
                            niche_count += share_function/50 #estimating a possible logic value for normalization--> popsize
                        elif distance == 0:
                            niche_count += 1
                        else:
                            share_function = 0


                    print(niche_count)
                    offsprings[j].fitness = offsprings[j].fitness/niche_count

            fitnesses.append(elite_offspring.fitness)


            if report:
                self._verbose_reporter_inner(elite, iteration)

            if log:
                log_event = [iteration, elite.fitness, elite.validation_fitness if hasattr(off2, 'validation_fitness') else None,
                             self.population_size, self.selection.__name__, self.crossover.__name__, self.p_c,
                             self.mutation.__name__, None, None, self.p_m, self._phenotypic_diversity_shift(offsprings)]
                logger.info(','.join(list(map(str, log_event))))

            self.population = offsprings

            if(iteration==n_iterations-1):
                plt.plot(fitnesses)

        self.best_solution = elite
        print(self.best_solution.fitness)
        uls.save_object(self.best_solution.representation,'/baseline/representation.pkl')

    def _crossover(self, p1, p2):
        off1, off2 = self.crossover(p1.representation, p2.representation, self._random_state)
        off1, off2 = Solution(off1), Solution(off2)
        return off1, off2

    def _mutation(self, individual):
        mutant = self.mutation(individual.representation, self._random_state)
        mutant = Solution(mutant)
        return mutant

    def _get_elite(self, population):
        elite = reduce(self._get_best, population)
        return elite

    def _phenotypic_diversity_shift(self, offsprings): #negative if the new generation has less "variety" in a phenotypical way
        fitness_parents = np.array([parent.fitness for parent in self.population])
        fitness_offsprings = np.array([offspring.fitness for offspring in offsprings])
        return np.std(fitness_offsprings)-np.std(fitness_parents)

    def _generate_random_valid_solutions(self):
        solutions = np.array([self._generate_random_valid_solution()
                              for i in range(self.population_size)])
        return solutions