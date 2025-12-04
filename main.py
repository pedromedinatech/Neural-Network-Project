from src.experiment import run_experiment 
import matplotlib.pyplot as plt
import os

def plot_and_save(test_data, train_data, historical_fitness, config, exp):

    file_name = (f"exp_{exp}.png")

    file_path = os.path.join("results", file_name)

    plt.figure(figsize=(12, 7))

    plt.plot(historical_fitness)

    plt.title(f"Experiment {exp}: Fitness Evolution\n"
              f"Architecture: {config['architecture']} | Population: {config['population']} | Gens: {config['max_generations']} | Mut: {config['mutation_rate']}\n"
              f"Train Acc: {train_data:.2f}% | Test Acc: {test_data:.2f}%", fontsize=10)
    
    plt.xlabel("Generation")
    
    plt.ylabel("Best Fitness")

    plt.grid(True)
    plt.ylim(0.0, 1.0)

    plt.savefig(file_path)

def main():

    experiments_list = [

        {"architecture" : [30, 15, 1], "population" : 10, "max_generations" : 50, "crossover_rate" : 0.8, "mutation_rate" : 0.2},
        {"architecture" : [30, 20, 10, 1], "population" : 100, "max_generations" : 300, "crossover_rate" : 0.8, "mutation_rate" : 0.2},
        {"architecture": [30, 50, 1], "population": 100, "max_generations": 300, "crossover_rate": 0.8, "mutation_rate": 0.2},
        {"architecture": [30, 10, 10, 5, 1], "population": 100, "max_generations": 300, "crossover_rate": 0.8, "mutation_rate": 0.2},
    ]
    
    """
    I am wondering to not tweak the mutation/crossover rates at the beginning.
    The architecture and population size seem to be the most relevant parameters.
    Maybe it is a good idea to start tweaking them inside the experiment, when
    the EA is running and the model seem to be stuck. From the first version,
    we can see the model tends to stagnate after some generations. So maybe it is a 
    great idea to let it do crossover, and when stagnation is detected, increase mutation rate
    for some generations, then set it back to the original value.

    So my approach will be, only tweak architecture, population size and max generations here.

    So we have to modify the EA class to allow dynamic mutation rate changes during execution.
    """
    for i, configuration in enumerate(experiments_list):

        print(f"Starting experiment {i+1}/{len(experiments_list)}\n")

        print(f"Configuration: {configuration}\n")

        test_acc, train_acc, historical_fitness = run_experiment(
                    MLP_ARCHITECTURE=configuration["architecture"],
                    POPULATION_SIZE=configuration["population"],
                    MAX_GENERATIONS=configuration["max_generations"],
                    CROSSOVER_RATE=configuration["crossover_rate"],
                    MUTATION_RATE=configuration["mutation_rate"])


        plot_and_save(test_acc, train_acc, historical_fitness, configuration, i)

if __name__ == "__main__":

    main()