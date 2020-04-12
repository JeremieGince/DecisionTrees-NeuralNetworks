from classifieur import *
from typing import *
import numpy as np
from random import random
from math import e


class activationFunction:
    def __init__(self):
        return

    def getValue(self, activation: float) -> float:
        raise NotImplementedError()


class hardThreshold(activationFunction):
    threshold: float = 0.0

    def __init__(self, p_threshold: float = 0.0):
        super().__init__()
        self.threshold = p_threshold

    def getValue(self, activation: float) -> float:
        if activation < self.threshold:
            return 0.0
        else:
            return 1.0


class logisticFunction(activationFunction):
    def __init__(self):
        super().__init__()

    def getValue(self, activation: float) -> float:
        return 1.0 / (1.0 + e ** ((-1.0) * activation))


class logisticFunction_derivative(activationFunction):
    def __init__(self):
        super().__init__()

    def getValue(self, activation: float) -> float:
        v: float = logisticFunction().getValue(activation)
        return v * (1 - v)


"""
class hardThreshold_derivative(activationFunction):
    threshold: float = 0.0

    def __init__(self, p_threshold: float = 0.0):
        super().__init__()
        self.threshold = p_threshold

    def getValue(self, activation: float) -> float:
        if activation < self.threshold:
            return 0.0
        else:
            return 1.0
"""


class Layer:
    activation_function: activationFunction = activationFunction()
    activation_function_derivative: activationFunction = activationFunction()
    incoming_weights = np.array([])
    last_activation: np.array = np.array([])
    last_in: np.array = 0.0
    n = 0

    def __init__(self, p_activation_function, p_activation_function_derivative, p_incoming_weights: np.array):
        self.activation_function = p_activation_function
        self.activation_function_derivative = p_activation_function_derivative
        self.incoming_weights = p_incoming_weights

    def _getNumberOfNeurons(self) -> int:
        return self.incoming_weights.shape[0]

    def propagate(self, activation: np.array) -> np.array:
        number_of_neurons: int = self._getNumberOfNeurons()
        self.last_in = activation.copy()
        propagator: np.array = np.array([0.0 for i in range(number_of_neurons)])
        for i in range(number_of_neurons):
            value: float = np.dot(activation, self.incoming_weights[i,])
            propagator[i] = self.activation_function.getValue(value)
        self.last_activation = propagator.copy()
        return propagator

    def backPropagate(self, errors: np.array, alpha: float) -> np.array:
        # self._adjustWeights(errors, alpha)
        derivative: np.array = self.applyDerivative()
        number_links_to: int = self.incoming_weights.shape[1]
        propagator: np.array = np.array([0.0 for i in range(self._getNumberOfNeurons())])
        for i in range(number_links_to):
            sum = 0.0
            for j in range(self._getNumberOfNeurons()):
                sum += self.incoming_weights[j, i] * errors[j]
            propagator[i] = sum*derivative[i]
        return propagator

    def _adjustWeights(self, error: np.array, alpha: float):
        if error.size != self.last_activation.size:
            raise RuntimeError("Something is wrong in _adjustWeights")
        for i in range(self.last_activation.size):
            self.incoming_weights[i,] += self.last_activation[i] * alpha * error

    def applyDerivative(self) -> np.array:
        return np.array(
            [self.activation_function_derivative.getValue(self.last_in[i]) for i in range(self.last_in.size)])

    def applyDerivative_out(self) -> np.array:
        return np.array([self.activation_function_derivative.getValue(self.last_activation[i]) for i in
                         range(self.last_activation.size)])

    """
    def backPrpagate(self, error: np.array) -> np.array:
        number_of_neurons: int = self._getNumberOfNeurons()
        propagator: np.array = np.array([0.0 for i in range(number_of_neurons)])
        for i in range(number_of_neurons):
            propagator[i] = self.activation_function.getValue(np.dot(error, self.incoming_weights[i,]))
        self.last_activation = propagator.copy()
        return propagator
    """


class NeuralNet(Classifier):
    number_of_layers: int = 0
    number_of_neurons_per_layer: int = 0
    layers: dict = dict()
    learning_rate: float = 0.001
    activation_function: activationFunction = logisticFunction()
    activation_function_derivative: activation_function = logisticFunction_derivative()
    initializeWeightsWithValue: float = random()  # between 0 and 1

    def __init__(self, p_number_of_layers: int, p_number_of_neurons_per_layer: int, **kwargs):
        super().__init__(**kwargs)
        self.number_of_layers = p_number_of_layers
        self.number_of_neurons_per_layer = p_number_of_neurons_per_layer
        self._initializeHiddenLayers()
        self._initializeInputLayer(3)
        self._initializeOutputLayer(3)
        v = self._propagate(np.array([-10, 0.001, 0.0001]))
        s = self._getInitialDelta(np.array([5.0, 4.0, 20.0]), np.array([5.0, 3.0, 15.0]))
        self._backPropagate(np.array([5.0, 4.0, 20.0]), np.array([5.0, 3.0, 15.0]))
        return

    def _makeWeightsData(self, number_neurons_from: int, number_of_neurons_to: int) -> np.array:
        one_neuron: list = [self.initializeWeightsWithValue for i in range(number_of_neurons_to)]
        all_links: list = [one_neuron for i in range(number_neurons_from)]
        return np.array(all_links)

    def _getCurrentNumberOfLayers(self):
        return len(self.layers)

    def _initializeHiddenLayers(self):
        for i in range(1, self.number_of_layers):
            self.layers[i] = Layer(self.activation_function, self.activation_function_derivative,
                                   self._makeWeightsData(self.number_of_neurons_per_layer,
                                                         self.number_of_neurons_per_layer))

    # input layer is really just the first layer of hidden layers
    def _initializeInputLayer(self, number_of_features: int):
        self.layers[0] = Layer(self.activation_function, self.activation_function_derivative,
                               self._makeWeightsData(self.number_of_neurons_per_layer, number_of_features))

    def _initializeOutputLayer(self, number_of_classes: int):
        self.layers[self._getCurrentNumberOfLayers()] = Layer(self.activation_function,
                                                              self.activation_function_derivative,
                                                              self._makeWeightsData(number_of_classes,
                                                                                    self.number_of_neurons_per_layer))

    def _propagate(self, feature_vector: np.array) -> np.array:
        current_activations: np.array = feature_vector.copy()
        for i in range(self._getCurrentNumberOfLayers()):
            current_activations = self.layers[i].propagate(current_activations)
        return current_activations

    def _getInitialDelta(self, output: np.array, actuals: np.array) -> np.array:
        derivative_of_output_layer: np.array = self.layers[self._getCurrentNumberOfLayers() - 1].applyDerivative_out()
        y_minus_a: np.array = np.array([0.0 for i in range(output.size)], dtype=float)
        np.subtract(actuals, output, y_minus_a)
        to_return: np.array = np.array([0.0 for i in range(y_minus_a.size)], dtype=float)
        np.multiply(derivative_of_output_layer, y_minus_a, to_return)
        return to_return

    def _backPropagate(self, output: np.array, actuals: np.array):
        number_of_layers: int = self._getCurrentNumberOfLayers()
        initial_error: np.array = self._getInitialDelta(output, actuals)
        current_err: np.array = initial_error.copy()
        for i in range(number_of_layers-1, -1, -1):
            current_err = self.layers[i].backPropagate(current_err, self.learning_rate)

    def train(self, train_set: np.ndarray, train_labels: np.ndarray,
              verbose: bool = True, **kwargs):
        """
        Used to train_set the current model.

        :param train_set: une matrice de type Numpy et de taille nxm, avec
        n : le nombre d'example d'entrainement dans le dataset
        m : le mobre d'attribus (le nombre de caractéristiques)

        :param train_labels : est une matrice numpy de taille nx1
        :param verbose: True To print results else False. (bool)
        :param kwargs: Parameters to pass to child classes.
        """
        super().train(train_set, train_labels, verbose)

        raise NotImplementedError()


if __name__ == "__main__":
    NeuralNet(5, 5)
