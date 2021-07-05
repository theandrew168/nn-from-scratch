import argparse

import numpy as np
from tqdm import tqdm


class NeuralNetwork:

    def __init__(self, num_input, num_hidden, num_output, learning_rate):
        self.num_input = num_input
        self.num_hidden = num_hidden
        self.num_output = num_output
        self.learning_rate = learning_rate

        self.weights_ih = np.random.normal(
            0.0,
            pow(self.num_hidden, -0.5),
            (self.num_hidden, self.num_input))
        self.weights_ho = np.random.normal(
            0.0,
            pow(self.num_output, -0.5),
            (self.num_output, self.num_hidden))

        self.activation_function = self.sigmoid

    def train(self, inputs, targets):
        inputs = np.array(inputs, ndmin=2).T
        targets = np.array(targets, ndmin=2).T

        hidden_inputs = np.dot(self.weights_ih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = np.dot(self.weights_ho, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        output_errors = targets - final_outputs
        hidden_errors = np.dot(self.weights_ho.T, output_errors)

        self.weights_ho += self.learning_rate * np.dot(
            output_errors * final_outputs * (1.0 - final_outputs),
            hidden_outputs.T)
        self.weights_ih += self.learning_rate * np.dot(
            hidden_errors * hidden_outputs * (1.0 - hidden_outputs),
            inputs.T)

    def query(self, inputs):
        inputs = np.array(inputs, ndmin=2).T

        hidden_inputs = np.dot(self.weights_ih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = np.dot(self.weights_ho, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        return final_outputs

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train and query the MNIST dataset')
    parser.add_argument('--epochs', type=int, default=1, help='number of training iterations')
    parser.add_argument('--hidden-nodes', type=int, default=100, help='number of hidden nodes')
    parser.add_argument('--learning-rate', type=float, default=0.3, help='network learning rate')
    parser.add_argument('--train', type=str, default='mnist_train.csv', help='dataset to use for training')
    parser.add_argument('--query', type=str, default='mnist_test.csv', help='dataset to use for querying')
    args = parser.parse_args()

    # create the initial network
    input_nodes = 784
    output_nodes = 10
    nn = NeuralNetwork(input_nodes, args.hidden_nodes, output_nodes, args.learning_rate)

    # train the network
    with open(args.train) as f:
        lines = f.readlines()

    for epoch in range(args.epochs):
        print('epoch {} of {}'.format(epoch + 1, args.epochs))
        for line in tqdm(lines, desc='training', unit=' rows'):
            number, *pixels = line.split(',')
            number = int(number)
            inputs = (np.asfarray(pixels) / 255.0 * 0.99) + 0.01

            targets = np.zeros(output_nodes) + 0.01
            targets[number] = 0.99

            nn.train(inputs, targets)

    # query the network
    with open(args.query) as f:
        lines = f.readlines()

    scores = []
    for line in tqdm(lines, desc='querying', unit=' rows'):
        number, *pixels = line.split(',')
        number = int(number)
        inputs = (np.asfarray(pixels) / 255.0 * 0.99) + 0.01

        outputs = nn.query(inputs)
        guess = np.argmax(outputs)

        if guess == number:
            scores.append(1)
        else:
            scores.append(0)

    # check how well the network performed
    print('accuracy:', sum(scores) / len(scores))
