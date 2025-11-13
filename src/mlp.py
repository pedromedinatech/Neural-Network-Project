import numpy as np

class MLP():

    def __init__(self, layer_sizes):
        
        self.layer_sizes = layer_sizes
        self.weights = []
        self.biases = []
        self.total_parameters = self.get_total_parameters()

    # Since the activation functions don't use self,
    # they can be static. 

    @staticmethod
    def ReLu(x):

        """
        Applies the Rectified Linear Unit (ReLU) activation function.
        """
        return np.maximum(0, x)

    @staticmethod
    def sigmoid(x):

        """
        Applies the Sigmoid activation function.
        Used for the final output layer in binary classification.
        """
        # Clip x to prevent overflow in np.exp()
        x_clipped = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x_clipped))


    def get_total_parameters(self):
        
        """
        Calculates the total number of weights and biases needed
        for the architecture specified in self.layer_sizes. This 
        will be our number of chromosomes for the EA.

        For a better clarification, the self.layer_size attribute
        looks like this:

        [30, 20, 10, 1], making reference to the number of neurons in each layer.

        As the network structure can vary, we have to generalize the method for
        obtaining the parameters that need to be processed.
        """

        total_params = 0

        for i in range(len(self.layer_sizes) - 1):
            
            # Get the size of the input layer and the output layer
            # for this specific connection.

            input_size = self.layer_sizes[i]
            output_size = self.layer_sizes[i + 1]

            total_params += (input_size * output_size) + output_size
            
        return total_params
    
    def set_from_chromosome(self, genes_list):

        """
        This method will "un-flatten" the 1D gene list from the chromosome
        and build the actual weight and bias matrices.

        At the end of the function execution, self.weights and self.biases
        will contain all the corresponding parameters of the neural network.
        """
        self.weights = []
        self.biases = []

        # Did this to ensure we use np arrays throughout the project
        genes_list = np.array(genes_list)

        # I create a pointer here for iterating 
        # through the genes_list. 
        idx_pointer = 0

        for i in range(len(self.layer_sizes) - 1):

            input_size = self.layer_sizes[i]
            output_size = self.layer_sizes[i + 1]

            # I operate the weights and biases separately, and
            # add them to self.weights and self.biases respectively

            n_weights = input_size * output_size

            weight_slice = genes_list[idx_pointer : idx_pointer + n_weights]
            weight_matrix = weight_slice.reshape(input_size, output_size)
            self.weights.append(weight_matrix)

            idx_pointer += n_weights

            n_biases = output_size

            bias_slice = genes_list[idx_pointer : idx_pointer + n_biases]
            bias_vector = bias_slice.reshape(1, n_biases)
            self.biases.append(bias_vector)

            idx_pointer += n_biases


            
    def feed_forward(self, inputs):

        """
        This method will take a numpy array of inputs,
        pass it through the network, and return the final output.
        """
        # I do this to ensure np.dot works correctly
        current_activation = inputs.reshape(1, self.layer_sizes[0])

        for i in range(len(self.weights) - 1):

            w = self.weights[i]
            b = self.biases[i]

            z = np.dot(current_activation, w) + b

            current_activation = self.ReLu(z)

        w_output = self.weights[-1]
        b_output = self.biases[-1]

        z_output = np.dot(current_activation, w_output) + b_output

        final_prediction = self.sigmoid(z_output)

        return final_prediction