import numpy as np

from evolutionary_algorithm import EvolutionaryAlgorithm
from cancer_problem import BreastCancerProblem

def main():

    #MLP_ARCHITECTURE = [30, 20, 10, 1]

    MLP_ARCHITECTURE = [30, 15, 1]

    # MLP_ARCHITECTURE = [30, 25, 15, 5, 1]
    """
    To make the implementation simpler,
    I hard-coded the architecture but can change
    it quickly by removing the comment.

    """
    # Start by defining the parameters for the EA.
    # Solve receives: problem, population_size, max_generations, crossover_rate and mutation_rate

    POPULATION_SIZE = 100 # Standard value 
    MAX_GENERATIONS = 50 # Temporary value, we can adjust according to the fitness
    CROSSOVER_RATE = 0.8
    MUTATION_RATE = 0.2

    print(f"Architecture: {MLP_ARCHITECTURE}")
    print(f"Population: {POPULATION_SIZE} | Generations: {MAX_GENERATIONS}\n")

    print("Loading and preprocessing data...")
    problem = BreastCancerProblem(MLP_ARCHITECTURE)
    print(f"Data loaded. Chromosome length: {problem.no_genes} genes.")

    # In the documentation, a further explanation of the hyperparameters will be done.
    # Initial ones: POPULATION_SIZE = 100, MAX_GENERATIONS = 500, CROSSOVER_RATE = 0.8 and MUTATION_RATE = 0.2

    ea = EvolutionaryAlgorithm()

    solution = ea.solve(problem, 
                        POPULATION_SIZE, 
                        MAX_GENERATIONS, 
                        CROSSOVER_RATE, 
                        MUTATION_RATE)
    """
    Here, the program may seem stuck. PyTorch implementation is way faster,
    but as we are using pure numpy, the operations may take longer.

    (50,000 fitness evaluations) * (454 samples per evaluation) ≈ 22,700,000 
    Those are a lot of feed_forward calls, which are processing matrix multiplications
    """
    print("\n--- Final Results ---")
    print(f"Best Training Fitness (Accuracy): {solution.fitness * 100:.2f}%")

    # Configure our reusable MLP with the best genes found
    final_mlp = problem.mlp_evaluator
    final_mlp.set_from_chromosome(solution.genes)

    total_correct_test = 0
    total_test_samples = len(problem.X_test_scaled)

    for i in range(total_test_samples):

        sample_x = problem.X_test_scaled[i]
        y_label = problem.y_test[i]

        prediction = final_mlp.feed_forward(sample_x)
        prediction = np.round(prediction)

        if prediction == y_label:

            total_correct_test += 1

    test_accuracy = total_correct_test / total_test_samples

    print(f"Final Test Accuracy (Generalization): {test_accuracy * 100:.2f}%")

if __name__ == "__main__":

    main()