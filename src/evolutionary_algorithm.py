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
        
        # Randomly selects two different indices in population

        indexes = random.sample(range(len(population)), 2)

        index1, index2 = indexes[0], indexes[1]
        
        # Retrieve both candidate chromosomes

        chromosome1 = population[index1]
        chromosome2 = population[index2]

        # Compares fitness values
        # Returns a copy with the best one

        if chromosome1.fitness > chromosome2.fitness:
            return chromosome1.__copy__()
        else:
            return chromosome2.__copy__()
        


    @staticmethod
    def get_best(population):
        
        # Iterates through population and finds the index of the chromosome with maximum fitness.
        # Create and returns a new instance/copy of that chromosome (not the original reference).

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
        
        # Applies arithmetic crossing between mother and father with p = rate

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

        historical_fitness = []

        best_fitness_ever = 0.0
        generations_without_improvement = 0

        stagnation_threshold = 50  # Number of generations to consider for stagnation

        for gen in range(max_generations):

            current_best = Selection.get_best(population)

            current_fitness = current_best.fitness

            if current_fitness > best_fitness_ever:
                best_fitness_ever = current_fitness
                generations_without_improvement = 0
                current_mutation_rate = mutation_rate
            else:
                generations_without_improvement += 1

            if generations_without_improvement >= stagnation_threshold:

                current_mutation_rate = 0.5  # Increase mutation rate to 50% during stagnation
            else:
                current_mutation_rate = mutation_rate

            new_population = [current_best]

            for i in range(1, population_size):
                # select 2 parents: Selection.tournament
                father = Selection.tournament(population)
                mother = Selection.tournament(population)

                # generate a child by applying crossover: Crossover.arithmetic
                child = Crossover.arithmetic(father, mother, crossover_rate)

                # apply mutation to the child: Mutation.reset
                Mutation.reset(child, current_mutation_rate)

                # calculate fitness for the child: compute_fitness from problem p
                problem.compute_fitness(child)

                # insert the child into new_population
                new_population.append(child)

            population = new_population
            best_fitness_so_far = population[0].fitness
            historical_fitness.append(best_fitness_so_far)
            print(f"Generation {gen+1}/{max_generations}, Best Fitness: {best_fitness_so_far * 100:.2f}%", end="\r")

        return Selection.get_best(population), historical_fitness
