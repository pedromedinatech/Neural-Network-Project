import math 
import random

class Chromosome:
    def __init__(self, no_genes, min_values, max_values):

        self.no_genes = no_genes
        self.fitness = 0.0
        
        """
        I had to adapt this method in respect of the Lab 8 implementation,
        since the problem does not pass a list of min/max values, but a value
        """
        if isinstance(min_values, (int, float)):
            # If it is, create a list by repeating it no_genes times
            self.min_values = [min_values] * no_genes
        else:
            # Otherwise, assume it's already a list and convert it
            self.min_values = list(min_values)

        # Do the same check for max_values
        if isinstance(max_values, (int, float)):
            self.max_values = [max_values] * no_genes
        else:
            self.max_values = list(max_values)

        self.genes = [random.uniform(self.min_values[i], self.max_values[i]) 
                      for i in range(no_genes)]
        

    def _initialize_genes(self):
        for i in range(self.no_genes):
            self.genes[i] = self.min_values[i] + random.random() * (self.max_values[i] - self.min_values[i])

    def __copy__(self):
        new_copy = Chromosome(self.no_genes, self.min_values, self.max_values)
        new_copy.genes = list(self.genes)
        new_copy.fitness = self.fitness
        return new_copy

    def copy_from(self, other):
        self.no_genes = other.no_genes
        self.genes = list(other.genes)
        self.min_values = list(other.min_values)
        self.max_values = list(other.max_values)
        self.fitness = other.fitness


class Selection:

    @staticmethod
    def tournament(population):
        
        # seleccionar aleatoriamente dos indices distintos en population

        indexes = random.sample(range(len(population)), 2)

        index1, index2 = indexes[0], indexes[1]
        
        # recuperar los dos cromosomas candidatos

        chromosome1 = population[index1]
        chromosome2 = population[index2]

        # comparar sus valores de fitness (mas alto = mejor)
        # devuelve una copia nueva del cromosoma con mayor fitness

        if chromosome1.fitness > chromosome2.fitness:
            return chromosome1.__copy__()
        else:
            return chromosome2.__copy__()
        


    @staticmethod
    def get_best(population):
        
        # Itera toda la población y localiza el índice del cromosoma con fitness máximo.
        # Crea y devuelve una nueva instancia/copia de ese cromosoma (no la referencia original).
        # Razonamiento: importante para poder examinar o archivar el mejor individuo sin que luego la población lo modifique.

        best_fitness = population[0].fitness
        best_index = 0

        for i in range(1, len(population)):

            if population[i].fitness > best_fitness:

                best_fitness = population[i].fitness
                best_index = i

        best = population[best_index]

        return best.__copy__()


class Crossover:
    @staticmethod
    def arithmetic(mother, father, rate):
        
        # Aplica el cruce aritmético entre los padres, usando probabilidad rate

        child = Chromosome(mother.no_genes, mother.min_values, mother.max_values)

        # Rate is the p of the crossover happening

        if random.random() < rate:

            # alpha = random.randint(0, 1)
            # randint generates an integer
         
            # alpha should be a real value between 0 and 1
            alpha = random.random()

            for i in range(mother.no_genes):
                
                child.genes[i] = alpha * mother.genes[i] + (1.0 - alpha) * father.genes[i]
        
        else:

            # If we don't do crossover, we give the same probability to the child heir the parent's gene
            if random.random() < 0.5:

                child.genes = list(mother.genes)

            else:

                child.genes = list(father.genes)

        return child


class Mutation:
    @staticmethod
    def reset(child, rate):
        
        # Apply the mutation to the child reseting it with p = rate

        # We need mutation so the range of possible values we find can be found
        # in any interval. Otherwise, the intervals the genes are will always be
        # a linear combination of their parent's genes.

        for i in range(child.no_genes):

            if random.random() < rate:
                
                min_val = child.min_values[i]

                max_val = child.max_values[i]

                child.genes[i] = min_val + random.random() * (max_val - min_val)

class EvolutionaryAlgorithm:
    def solve(self, problem, population_size, max_generations, crossover_rate, mutation_rate):

        population = [problem.make_chromosome() for _ in range(population_size)]
        for individual in population:
            problem.compute_fitness(individual)

        for gen in range(max_generations):
            new_population = [Selection.get_best(population)]

            for i in range(1, population_size):
                # select 2 parents: Selection.tournament
                father = Selection.tournament(population)
                mother = Selection.tournament(population)

                # generate a child by applying crossover: Crossover.arithmetic
                child = Crossover.arithmetic(father, mother, crossover_rate)

                # apply mutation to the child: Mutation.reset
                Mutation.reset(child, mutation_rate)

                # calculate fitness for the child: compute_fitness from problem p
                problem.compute_fitness(child)

                # insert the child into new_population
                new_population.append(child)

            population = new_population
            best_fitness_so_far = population[0].fitness
            print(f"Generation {gen+1}/{max_generations}, Best Fitness: {best_fitness_so_far * 100:.2f}%", end="\r")

        return Selection.get_best(population)
