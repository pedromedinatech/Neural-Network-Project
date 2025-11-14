import numpy as np

from .evolutionary_algorithm import EvolutionaryAlgorithm
from .cancer_problem import BreastCancerProblem

def run_experiment(MLP_ARCHITECTURE, POPULATION_SIZE, MAX_GENERATIONS, CROSSOVER_RATE, MUTATION_RATE):

    """
    Executes a complete simulation for the EA for the hyperparameters given.

    Returns the historical fitness, the precision for testing and the training accuracy.
    """
    print(f"Running Experiment for:")
    print(f"Architecture: {MLP_ARCHITECTURE}\n")
    print(f"Population: {POPULATION_SIZE}\n")
    print(f"Generations: {MAX_GENERATIONS}\n")

    print("Loading and preprocessing data...")
    problem = BreastCancerProblem(MLP_ARCHITECTURE)
    print(f"Data loaded correctly. Chromosome length: {problem.no_genes} genes.")

    # In the documentation, a further explanation of the hyperparameters will be done.
    # Initial ones: POPULATION_SIZE = 100, MAX_GENERATIONS = 500, CROSSOVER_RATE = 0.8 and MUTATION_RATE = 0.2

    ea = EvolutionaryAlgorithm()

    solution, historical_fitness = ea.solve(problem, 
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
    training_precision = solution.fitness * 100

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

    test_accuracy = (total_correct_test / total_test_samples) * 100

    return test_accuracy, training_precision, historical_fitness

