from classifieur import *
from typing import *
import numpy as np
from random import random


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
            return 0
        else:
            return 1


class Layer:
    activation_function: activationFunction = activationFunction()
    incoming_weights = np.array([])
    n = 0

    def __init__(self, p_activation_function, p_incoming_weights: np.array):
        self.activation_function = p_activation_function
        self.incoming_weights = p_incoming_weights

    def applyAndGetActivation(self, activation: np.array, neurone_number: int) -> float:
        return self.activation_function.getValue(np.dot(activation, self.incoming_weights[neurone_number,]))

    def getNumberOfNeurons(self):
        return self.incoming_weights.shape[0]


class NeuralNet(Classifier):
    number_of_layers: int = 0
    number_of_neurons_per_layer: int = 0
    layers: dict = dict()
    activation_function: activationFunction = hardThreshold()
    initializeWeightsWithValue: float = random()  # between 0 and 1

    def __init__(self, p_number_of_layers: int, p_number_of_neurons_per_layer: int, **kwargs):
        super().__init__(**kwargs)
        self.number_of_layers = p_number_of_layers
        self.number_of_neurons_per_layer = p_number_of_neurons_per_layer
        self._initializeHiddenLayers()
        self._initializeInputLayer(3)
        self._initializeOutputLayer(2)
        v = self._propagate(np.array([1.0, 2.0, 3.0]))
        return

    def _makeWeightsData(self, number_neurons_from: int, number_of_neurons_to: int) -> np.array:
        one_neuron: list = [self.initializeWeightsWithValue for i in range(number_of_neurons_to)]
        all_links: list = [one_neuron for i in range(number_neurons_from)]
        return np.array(all_links)

    def _getCurrentNumberOfLayers(self):
        return len(self.layers)

    def _initializeHiddenLayers(self):
        for i in range(1, self.number_of_layers):
            self.layers[i] = Layer(self.activation_function, self._makeWeightsData(self.number_of_neurons_per_layer,
                                                                                   self.number_of_neurons_per_layer))

    # input layer is really just the first layer of hidden layers
    def _initializeInputLayer(self, number_of_features: int):
        self.layers[0] = Layer(self.activation_function,
                               self._makeWeightsData(self.number_of_neurons_per_layer, number_of_features))

    def _initializeOutputLayer(self, number_of_classes: int):
        self.layers[self._getCurrentNumberOfLayers()] = Layer(self.activation_function,
                                                              self._makeWeightsData(number_of_classes,
                                                                                    self.number_of_neurons_per_layer))

    def _propagate(self, feature_vector: np.array) -> np.array:
        propagator: np.array = np.array([1.0 for i in range(self.number_of_neurons_per_layer)])
        current_activations: np.array = feature_vector
        for i in range(self.number_of_layers):
            for j in range(self.number_of_neurons_per_layer):
                propagator[j] = self.layers[i].applyAndGetActivation(current_activations, j)
            current_activations = propagator

        last_layer: Layer = self.layers[self._getCurrentNumberOfLayers() - 1]
        number_of_outputs: int = last_layer.getNumberOfNeurons()
        for k in range(number_of_outputs):
            propagator[k] = last_layer.applyAndGetActivation(current_activations, k)
        return propagator[:number_of_outputs]

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
