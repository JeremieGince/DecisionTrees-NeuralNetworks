from classifieur import *
from typing import *
import numpy as np
from math import e
from load_datasets import load_iris_dataset, load_congressional_dataset, load_monks_dataset
from random import random
import util
import matplotlib.pyplot as plt


def read_architecture_from_file(path: str):
    f = open(path, "r")
    lines = f.readlines()
    line = lines[0].rstrip()
    c = line.split(" ")
    f.close()
    return list(map(int, c))


def number_to_vector(value: int, total_number: int) -> np.array:
    ret: np.array = np.array([0.0 for i in range(total_number)], dtype=float)
    ret[value] = 1.0
    return ret


def to_prob_vector(a: np.array) -> np.array:
    to_return = np.array([0.0 for i in range(a.size)])
    to_return[np.argmax(a, axis=0)] = 1.0
    return to_return


def remove_minus_one(a: np.array):
    f = lambda x: x if x != -1.0 else 0.0
    return np.vectorize(f)(a)


def min_max_scale(a: np.array):
    for i in range(a.shape[0]):
        a[i,] = (a[i,] - a[i,].min()) / (a[i,].max() - a[i,].min())


class activationFunction:
    def __init__(self):
        return

    def getValue(self, activation: np.array):
        raise NotImplementedError()


class hardThreshold(activationFunction):
    threshold: float = 0.0

    def __init__(self, p_threshold: float = 0.0):
        super().__init__()
        self.threshold = p_threshold

    def getValue(self, activation: np.array):
        if activation < self.threshold:
            return 0.0
        else:
            return 1.0


class logisticFunction(activationFunction):
    def __init__(self):
        super().__init__()

    def getValue(self, activation: np.array):
        logistic = lambda x: 1.0 / (1.0 + e ** ((-1.0) * x))
        return logistic(activation)


class logisticFunction_derivative(activationFunction):
    def __init__(self):
        super().__init__()

    def getValue(self, activation: np.array):
        # v: float = logisticFunction().getValue(activation)
        logistic = lambda x: 1.0 / (1.0 + e ** ((-1.0) * x))
        return activation * (1 - activation)


class relu(activationFunction):
    pente: float = 1.0

    def __init__(self, p_pente: float = 1.0):
        super().__init__()
        self.pente = p_pente

    def getValue(self, activation: np.array):
        f = lambda x: max(x, 0) * self.pente
        return np.vectorize(f)(activation)


class relu_derivative(activationFunction):
    pente: float = 1.0

    def __init__(self, p_pente: float = 1.0):
        super().__init__()
        self.pente = p_pente

    def getValue(self, activation: np.array):
        f = lambda x: self.pente if x > 0.0 else 0.0
        return np.vectorize(f)(activation)


class softmax(activationFunction):
    def __init__(self):
        super().__init__()

    def getValue(self, activation: np.array):
        s = sum(np.exp(activation))
        a = np.exp(activation)
        return a / s


class softmax_der(activationFunction):
    def __init__(self):
        super().__init__()

    def getValue(self, activation) -> np.array:
        v: float = softmax().getValue(activation)
        return softmax().getValue(activation) * (1 - softmax().getValue(activation))


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
    def __init__(self, p_activation_function, p_activation_function_derivative, p_incoming_weights: np.array, **kwargs):
        self.bias_weights = kwargs.get("bias", np.random.rand(p_incoming_weights.shape[0]))
        self.bias = [np.nan for i in range(p_incoming_weights.shape[0])]
        self.activation_function = p_activation_function
        self.activation_function_derivative = p_activation_function_derivative
        self.incoming_weights = p_incoming_weights
        self.number_of_bias = 0
        self.last_activation = np.array([0.0 for i in range(self._getNumberOfNeurons())])


    def _getNumberOfNeurons(self) -> int:
        return self.incoming_weights.shape[0]

    def propagate(self, activation: np.array) -> np.array:
        number_of_neurons: int = self._getNumberOfNeurons()
        self.last_in = activation.copy()
        propagator: np.array = np.zeros(number_of_neurons)
        for i in range(number_of_neurons):
            value: float = np.dot(activation, self.incoming_weights[i,])
            # self.last_activation[i] = value
            propagator[i] = value
        self.last_activation = self.activation_function.getValue(propagator - self.bias_weights)
        return self.last_activation.copy()

    def backPropagate(self, errors: np.array, alpha: float) -> np.array:
        derivative: np.array = self.applyDerivative()
        v = np.dot(self.incoming_weights.transpose(), errors)
        self._adjustWeights(errors, alpha)
        self._adjustBias(errors, alpha)
        return derivative * v.transpose()

    def _adjustWeights(self, error: np.array, alpha: float):
        if error.size != self.last_activation.size:
            raise RuntimeError("Something is wrong in _adjustWeights")
        self.incoming_weights += alpha * np.dot(self.last_in[:, np.newaxis], error[np.newaxis, :]).transpose()

    def _adjustBias(self, error, alpha):
        self.bias_weights -= alpha*error

    def applyDerivative(self) -> np.array:
        return self.activation_function_derivative.getValue(self.last_in)

    def applyDerivative_out(self) -> np.array:
        return self.activation_function_derivative.getValue(self.last_activation)

    def addBias(self, bias_values):
        self.bias_weights = bias_values
        return

    def setWeights(self, weights: np.array):
        self.incoming_weights = weights


def _makeWeightsData(number_neurons_from: int, number_of_neurons_to: int) -> np.array:
    #one_neuron: list = [100*random() for i in range(number_of_neurons_to)]
    all_links: list = [[random() for i in range(number_of_neurons_to)] for j in range(number_neurons_from)]
    return np.array(all_links)


class NeuralNet(Classifier):

    def __init__(self, p_number_of_layers: int = 2, p_number_of_neurons_per_layer: int = 2,
                 p_number_output: int = 2,
                 explicit_architecture: list = None, **kwargs):
        super().__init__(**kwargs)
        self.normalizing_vector = None
        self.number_of_output = p_number_output
        self.nbr_epoch = 100
        self.number_of_layers = p_number_of_layers
        self.number_of_neurons_per_layer = p_number_of_neurons_per_layer
        self.learning_rate: float = 1.0
        self.layers: dict = dict()
        self.initializeWeightsWithValue: float = 0.0  # between 0 and 1
        self.activation_function: activationFunction = logisticFunction()
        self.activation_function_derivative = logisticFunction_derivative()
        self.weights = kwargs.get("weights",None)
        self.bias = kwargs.get("bias", None)
        self.fully_init = False
        if explicit_architecture is not None:
            if self.weights is None:
                self._initializeHiddenLayers(explicit_architecture)
            else:
                self.putWeightsInLayer(self.weights)
            if self.bias is not None:
                self.putBiasInLayer(self.bias)
            self.fully_init = True
        return

    def putBiasInLayer(self, biases):
        count = 0
        for bs in biases:
            self.layers[count].addBias(bs)
            count += 1


    def putWeightsInLayer(self, w):
        count = 0
        for ws in w:
            self.layers[count].setWeights(ws)
            count += 1
    def _getCurrentNumberOfLayers(self):
        return len(self.layers)

    def _initializeHiddenLayers(self, architecture: list):
        count: int = 0
        for i in range(len(architecture) - 1):
            self.layers[count] = Layer(self.activation_function, self.activation_function_derivative,
                                       _makeWeightsData(architecture[i + 1],
                                                             architecture[i]))
            count += 1

    # input layer is really just the first layer of hidden layers
    def _initializeInputLayer(self, number_of_features: int):
        self.layers[0] = Layer(self.activation_function, self.activation_function_derivative,
                               _makeWeightsData(self.number_of_neurons_per_layer, number_of_features))

    def _initializeOutputLayer(self, number_of_classes: int):
        self.layers[self._getCurrentNumberOfLayers()] = Layer(softmax(),
                                                              softmax_der(),
                                                              _makeWeightsData(number_of_classes,
                                                                                    self.number_of_neurons_per_layer))

    def _propagate(self, feature_vector: np.array) -> np.array:
        current_activations: np.array = feature_vector.copy()
        for i in range(self._getCurrentNumberOfLayers()):
            current_activations = self.layers[i].propagate(current_activations)
        return current_activations

    def getInitialDelta(self, output: np.array, actuals: np.array) -> np.array:
        derivative_of_output_layer: np.array = self.layers[self._getCurrentNumberOfLayers() - 1].applyDerivative_out()
        y_minus_a: np.array = np.array([0.0 for i in range(output.size)], dtype=float)
        #truc = to_prob_vector(output)
        np.subtract(actuals, output, y_minus_a)
        to_return: np.array = np.array([0.0 for i in range(y_minus_a.size)], dtype=float)
        np.multiply(derivative_of_output_layer, y_minus_a, to_return)
        return to_return

    def _backPropagate(self, output: np.array, actuals: np.array):
        number_of_layers: int = self._getCurrentNumberOfLayers()
        initial_error: np.array = self.getInitialDelta(output, actuals)
        current_err: np.array = initial_error.copy()
        for i in range(number_of_layers - 1, -1, -1):
            current_err = self.layers[i].backPropagate(current_err, self.learning_rate)

    def train(self, train_set: np.ndarray, train_labels: np.ndarray,
              verbose: bool = True, **kwargs):

        # super().train(train_set, train_labels, verbose)
        self.normalizing_vector = train_set.max(axis=0)
        train_set = train_set / self.normalizing_vector
        number_of_features: int = train_set.shape[1]
        number_of_examples: int = train_labels.size
        unique = np.unique(train_labels)
        number_of_classes = self.number_of_output
        if not self.fully_init:
            archi: list = [number_of_features]
            archi.extend([self.number_of_neurons_per_layer for i in range(self.number_of_layers)])
            archi.append(self.number_of_output)
            self._initializeHiddenLayers(archi)
            # self._initializeInputLayer(number_of_features)
            # self._initializeOutputLayer(number_of_classes)
            self.fully_init = True
        for j in range(self.nbr_epoch):
            for i in range(number_of_examples):
                f = train_set[i,]
                a = train_labels[i]
                out = self._propagate(f)
                self._backPropagate(out, number_to_vector(a, number_of_classes))

    def test2(self, test_set, test_labels, verbose: bool = True, displayArgs: dict = None) \
            -> (np.ndarray, float, float, float):
        if test_set.shape[0] != test_labels.size:
            raise RuntimeError("There is a problem in test")
        to_return = np.array([False for i in range(test_labels.size)])
        preds = np.array([0 for i in range(test_labels.size)])
        for i in range(test_labels.size):
            pred, res = self.predict(test_set[i,], test_labels[i])
            to_return[i] = res
            preds[i] = pred
        return preds, to_return

    def predict(self, example, label) -> (int, bool):
        out = self._propagate(example/self.normalizing_vector)
        pred = np.argmax(out, axis=0)
        return pred, pred == label

    def constructFromFile(self, path):
        archi : list = read_architecture_from_file(path)
        self._initializeHiddenLayers(archi)
        f = open(path, "r")
        bias:list = []
        lines = f.readlines()
        del lines[0]
        stuff_removed = list(map(str.rstrip, lines))
        for data in stuff_removed:
            if data[0][0] == "#":
                data_without_diese = data[1:]
                splitted = list(map(float,data_without_diese.split(" ")))
                bias = splitted

            else:
                splitted = list(map(float, data.split(" ")))
                w = splitted[1:]
                self.layers[int(splitted[0])].setWeights(np.array([w]))
        self.fully_init = True
        self.layers[int(bias[0])].addBias(int(bias[1]), bias[2])
        f.close()
        return

    @staticmethod
    def get_best_number_of_hidden_neurone(train_set, train_label , plot_results = True):
        accs = []
        n_neurone = []
        k = 5
        k_split_train = np.array_split(train_set, k, axis=0)
        k_split_train_labels = np.array_split(train_label, k, axis=0)
        for i in range(4, 51):
            nn = NeuralNet(3, i, np.unique(train_label).size)
            for j in range(k - 1):
                nn.train(k_split_train[j], k_split_train_labels[j])
            _, a, _, _ = nn.test(k_split_train[k - 1], k_split_train_labels[k - 1], False)
            accs.append(1.0 - a/100.0)
            n_neurone.append(i)
        if plot_results:
            plt.plot(n_neurone, accs)
            plt.show()
        return n_neurone[accs.]




if __name__ == "__main__":
    import load_datasets
    from load_datasets import congressionalFeatures, CongressionalValue, MonksFeatures, IrisFeatures

    import time
    import warnings

    warnings.filterwarnings("ignore")

    train_ratio_nn: float = 0.7
    prn = 20  # number of training per training_size for the compute of the Learning curve

    confusionMatrixList: list = list()

    print(f"Train ratio: {train_ratio_nn}")
    print("\n")

    print('-' * 175)
    print(f"Iris dataset classification: \n")
    # TODO: comparison with https://scikit-learn.org/stable/modules/tree.html#classification
    startTime = time.time()

    iris_train, iris_train_labels, iris_test, iris_test_labels = load_datasets.load_iris_dataset(train_ratio_nn)
    n = NeuralNet.get_best_number_of_hidden_neurone(iris_train, iris_train_labels)
    iris_nn = NeuralNet(3, n, np.unique(iris_train_labels).size)
    iris_nn.plot_learning_curve(iris_train, iris_train_labels, iris_test, iris_test_labels, save_name="iris_NN",
                                prn=prn)
    iris_nn.train(iris_train, iris_train_labels)
    cm, _, _, _ = iris_nn.test(iris_test, iris_test_labels)

    endTime = time.time() - startTime

    confusionMatrixList.append(cm)


    print(f"\n --- Elapse time: {1_000 * endTime:.2f} ms --- \n")

    print('-' * 175)
    print(f"Congressional dataset classification: \n")

    # TODO: change the missing values of the dataset for the mean of the values associated
    #      with the attribute of the missing values

    startTime = time.time()

    cong_train, cong_train_labels, cong_test, cong_test_labels = load_datasets.load_congressional_dataset(
        train_ratio_nn)

    cong_train = util.replaceMissingValues(cong_train, CongressionalValue.MISSING_VALUE.value)
    cong_test = util.replaceMissingValues(cong_test, CongressionalValue.MISSING_VALUE.value)
    n_cong = NeuralNet.get_best_number_of_hidden_neurone(cong_train, cong_train_labels)
    cong_dt = NeuralNet(3, n_cong, np.unique(cong_train_labels).size)
    cong_dt.plot_learning_curve(cong_train, cong_train_labels, cong_test, cong_test_labels, save_name="cong_NN",
                                prn=prn)
    cong_dt.train(cong_train, cong_train_labels)
    cm, _, _, _ = cong_dt.test(cong_test, cong_test_labels)

    endTime = time.time() - startTime

    confusionMatrixList.append(cm)

    print(f"\n --- Elapse time: {1_000 * endTime:.2f} ms --- \n")

    print('-' * 175)
    for i in range(3):
        print(f"Monks({i + 1}) dataset classification: \n")
        startTime = time.time()

        monks_train, monks_train_labels, monks_test, monks_test_labels = load_datasets.load_monks_dataset(i + 1)
        n_monks_one = NeuralNet.get_best_number_of_hidden_neurone(monks_train, monks_train_labels)
        monks_dt = NeuralNet(3, n, np.unique(monks_train_labels).size)

        monks_dt.plot_learning_curve(monks_train, monks_train_labels,
                                     monks_test, monks_test_labels, save_name=f"monks{i + 1}_DT", prn=prn)

        monks_dt.train(monks_train, monks_train_labels)
        cm, _, _, _ = monks_dt.test(monks_test, monks_test_labels)

        endTime = time.time() - startTime

        confusionMatrixList.append(cm)



        print(f"\n --- Elapse time: {1_000 * endTime:.2f} ms --- \n")

        print('-' * 175)

    Tpr, Fpr = util.computeTprFprList(confusionMatrixList, flattenOutput=False)

    util.plotROCcurves(Tpr, Fpr, hmCurve=5, labels=["Iris", "Congressional", "Monks(1)", "Monks(2)", "Monks(3)"],
                       title="Decision Tree - ROC curve")

