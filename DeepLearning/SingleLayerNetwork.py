from numpy import exp, array, random, dot


class NeuralNetwork():
    def __init__(self):
        # seed the random number generator, so it generates the same numbers
        # every time the program is ran
        random.seed(1)

        # we model a single neuron, with 3 input connections and 1 output connection
        # we assign random weights to a 3x1 matrix, with values in the range -1 to 1
        # and a mean of 0
        self.synaptic_weights = 2 * random.random((3, 1)) - 1

    # the signmoid function, which describes an s shaped curve
    # we pass the weighted sum of the inputs through this function
    # to normalize them between 0 and 1
    def __sigmoid(self, x):
        return 1 / (1 + exp(-x))

    # gradient of sigmoid curve
    def __sigmoid_derivative(self, x):
        return x * (1 - x)

    def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
        for iteration in range(number_of_training_iterations):
            # pass the training set through our neural net
            output = self.predict(training_set_inputs)

            # calculate the error
            error = training_set_outputs - output

            # multiply the error by the input and again by the gradient of the
            # sigmoid curve
            adjustment = dot(training_set_inputs.T, error * self.__sigmoid_derivative(output))

            # adjust the weights
            self.synaptic_weights += adjustment

    def predict(self, inputs):
        # pass inputs through our neural network (our single neuron)
        print('i: ', inputs.shape, inputs)
        print('s: ', self.synaptic_weights.shape, self.synaptic_weights)
        dot_result = dot(inputs, self.synaptic_weights)
        return self.__sigmoid(dot_result)


# if __name__ = '__main__':

# initialize a single neuron neural network
neural_network = NeuralNetwork()

print('Random starting synaptic weights:')
print(neural_network.synaptic_weights)

# the training set. We have 4 examples, each consisting of 3 input values
# and 1 output value
training_set_inputs = array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
training_set_outputs = array([[0, 1, 1, 0]]).T

# train the neural network using a training set.
# Do it 10000 times and make small adjustments each time
neural_network.train(training_set_inputs, training_set_outputs, 10000)

print('New synaptic weights after training: ')
print(neural_network.synaptic_weights)

# test the neural network with a new situation
print('Considering new sitatuion [1, 0, 0] -> ?: ')
print(neural_network.predict(array([1, 0, 0])))
