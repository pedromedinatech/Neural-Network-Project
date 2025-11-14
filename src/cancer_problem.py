import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


from .mlp import MLP
from .evolutionary_algorithm import Chromosome

class BreastCancerProblem:

    """
    This method will act as the join between the generic EA and 
    the specific use-case for the MLP. 
    
    The functions of this class will be:
        - make_chromosome(): creates a chromosome.
        - compute_fitness(): evaluates a chromosome.
    """

    def __init__(self, network_architecture):

        self.architecture = network_architecture

        # We need an instance of the MLP to be used for evaluation
        # This one will be re-configured by compute_fitness every time.
        # This is a key optimization: we reuse this one object
        # thousands of times, instead of creating a new one for each
        # chromosome evaluation in compute_fitness().

        self.mlp_evaluator = MLP(self.architecture)
    
        self.no_genes = self.mlp_evaluator.get_total_parameters()

        # We process the data from the dataframe (from the PyTorch implementation)
        # Note that the path does not start with .. because we execute this from main.py
        # in the parent directory.
        dataframe = pd.read_csv("data/data.csv")

        dataframe.iloc[:, 1] = dataframe.iloc[:,1].map({'M': 1.0, 'B' : 0.0})
        dataframe.replace("", np.nan, inplace=True)
        column_names_to_check = dataframe.columns[1:32]
        # I drop the NaN data
        dataframe.dropna(subset=column_names_to_check, inplace=True)

        X = dataframe.iloc[:, 2:32].values  # Features
        Y = dataframe.iloc[:, 1].values   # Targets

        Y = Y.astype(float)

        X_train, X_test, y_train, y_test = train_test_split(X, Y, 
                                                    test_size=0.2, 
                                                    random_state=42, 
                                                    stratify=Y)
        
        # I use the StandardScaler() to normalize the data.
        # It ensures a more reliable training. 

        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        self.X_train_scaled = X_train_scaled
        self.y_train = y_train
        self.X_test_scaled = X_test_scaled
        self.y_test = y_test


    def make_chromosome(self):

        # I pass -1.0 and 1.0 as min/max values as we don't have a list.
        # This requires a change in the EA class implementation, to ensure it
        # can process float numbers.
        return Chromosome(self.no_genes, -1.0, 1.0)
    
    def compute_fitness(self, chromosome):

        genes_list = chromosome.genes

        self.mlp_evaluator.set_from_chromosome(genes_list)

        total_success = 0

        total_samples = len(self.X_train_scaled)

        for i in range(total_samples):

            sample_x = self.X_train_scaled[i]
            label_y = self.y_train[i]

            prediction = self.mlp_evaluator.feed_forward(sample_x)

            prediction = np.round(prediction)

            if prediction == label_y:  

                total_success += 1

        
        accuracy = total_success / total_samples

        chromosome.fitness = accuracy


