# Neuroevolution for Breast Cancer Classification

This project explores the application of **Neuroevolution**—the use of evolutionary algorithms to generate artificial neural networks—to the field of medical diagnosis.

## Problem Statement

The objective is to accurately classify breast tumor images as **Malignant** or **Benign** based on features computed from digitized images of a fine needle aspirate (FNA) of a breast mass. The project utilizes the **Wisconsin Diagnostic Breast Cancer (WDBC)** dataset.

## Solution Approach

While traditional neural networks are trained using gradient-based methods like Backpropagation (e.g., Adam, SGD), this project builds a **Multi-Layer Perceptron (MLP) from scratch** using only NumPy and optimizes it using a **Genetic Algorithm (GA)**.

The Evolutionary Algorithm mimics natural selection to "evolve" the weights and biases of the network over generations, selecting for individuals (networks) that maximize classification accuracy. This approach decouples the architecture from the learning method, demonstrating how stochastic optimization can replace calculus-based training.

## Directory Structure

* **data/**
  * `data.csv`: The WDBC dataset used for training and testing.
* **results/**
  * Stores the output graphs visualizing the fitness evolution for each experiment.
* **src/**
  * `main.py`: The project orchestrator. It defines the experimental configurations (hyperparameters) and runs the simulation suite.
  * `experiment.py`: Encapsulates the logic for running a single complete evolutionary simulation.
  * `evolutionary_algorithm.py`: A generic, problem-agnostic implementation of a Genetic Algorithm (Selection, Crossover, Mutation).
  * `cancer_problem.py`: The "glue" class that connects the generic EA to the specific domain. It handles data loading, preprocessing, and fitness evaluation.
  * `mlp.py`: A custom, configurable Neural Network class implemented purely in NumPy with ReLU and Sigmoid activation functions.

## Dependencies

* Python 3.x
* numpy
* pandas
* scikit-learn
* matplotlib

## Installation

1. Ensure you are in the project root directory (`AI-Project-NNs/`).
2. Install the required Python libraries:

```bash
pip install numpy pandas scikit-learn matplotlib
```

## Usage

```
python src/main.py

```

The system will:

1. Load and preprocess the dataset (normalization, train/test split).

2. Run a series of predefined experiments with varying architectures and hyperparameters.

3. Print the Training Fitness and Test Accuracy for each experiment to the console.

4. Generate and save evolution plots (PNG files) in the results/ directory.

## Configuration

This algorithm is designed to be modular. Different configurations are the input to the neuron 
so a visual comparison is shown in /results for each configuration. In order to see or change any
of the configurations, access the src/main.py file. Hyperparameters are: 

1. architecture: the list that defines the neural network structure. 
Example for [30, 20, 10, 1], it means the NN has 30 neurons in the input layer, two hidden layers with 20 an 10 neurons each, 
and only one neuron in the input layer.
2. population: the population size of the genetic algorithm
3. max_generations: the number of generations to evolve
4. crossover_rate: probability of crossover happening 
5. mutation_rate: probability of mutation happening

