import load_datasets
from load_datasets import congressionalFeatures, CongressionalValue, MonksFeatures, IrisFeatures
import time
import warnings
from DecisionTree import DecisionTree
from NeuralNet import NeuralNet, plot_RN_ZERO_RN_NON_ZERO
import util
import matplotlib.pyplot as plt
import numpy as np
import random

# The folder "Code" is marked as root

warnings.filterwarnings("ignore")
random.seed(1)


###################################################################################################################
#  Partie 1 - Decision Tree
print("##########################################################################################################")
print("                                    Partie 1 - Decision Tree                                              ")
print("########################################################################################################## \n")
###################################################################################################################


train_ratio_dt: float = 0.7
prn = 20  # number of training per training_size to compute the Learning curve

confusionMatrixListDT: list = list()

print(f"Decision Tree Train ratio: {train_ratio_dt}")
print("\n")

print('-' * 175)
print(f"Iris dataset classification: \n")
# TODO: comparison with https://scikit-learn.org/stable/modules/tree.html#classification
startTime = time.time()

iris_train, iris_train_labels, iris_test, iris_test_labels = load_datasets.load_iris_dataset(train_ratio_dt)
iris_dt = DecisionTree(IrisFeatures, name="Iris Decision Tree")

# We don't compute the learning curve here, cause the computation takes several minutes.
# But you yan can compute it ny running the file: DecisionTree.py
# For information purposes, we will still display the results of this calculation.
# iris_dt.plot_learning_curve(iris_train, iris_train_labels, iris_test, iris_test_labels, save_name="iris_DT", prn=prn)
plt.title("Learning curve of Decision Tree - Iris")
plt.imshow(plt.imread("Figures/Learning_curve_iris_DT.png"))
plt.show()
iris_dt.train(iris_train, iris_train_labels)
cm, _, _, _ = iris_dt.test(iris_test, iris_test_labels)

endTime = time.time() - startTime

confusionMatrixListDT.append(cm)

iris_dt.draw(save=False)

print(f"\n --- Elapse time: {1_000 * endTime:.2f} ms --- \n")

print('-' * 175)
print(f"Congressional dataset classification: \n")

startTime = time.time()

cong_train, cong_train_labels, cong_test, cong_test_labels = load_datasets.load_congressional_dataset(train_ratio_dt)

cong_train = util.replaceMissingValues(cong_train, CongressionalValue.MISSING_VALUE.value)
cong_test = util.replaceMissingValues(cong_test, CongressionalValue.MISSING_VALUE.value)

cong_dt = DecisionTree(congressionalFeatures, name="Congressional Decision Tree")

# We don't compute the learning curve here, cause the computation takes several minutes.
# But you yan can compute it ny running the file: DecisionTree.py
# For information purposes, we will still display the results of this calculation.
# cong_nn.plot_learning_curve(cong_train, cong_train_labels, cong_test, cong_test_labels, save_name="cong_DT", prn=prn)
plt.title("Learning curve of Decision Tree - Congressional")
plt.imshow(plt.imread("Figures/Learning_curve_cong_DT.png"))
plt.show()
cong_dt.train(cong_train, cong_train_labels)
cm, _, _, _ = cong_dt.test(cong_test, cong_test_labels)

endTime = time.time() - startTime

confusionMatrixListDT.append(cm)

cong_dt.draw(save=False)

print(f"\n --- Elapse time: {1_000 * endTime:.2f} ms --- \n")

print('-' * 175)
for i in range(3):
    print(f"Monks({i + 1}) dataset classification: \n")
    startTime = time.time()

    monks_train, monks_train_labels, monks_test, monks_test_labels = load_datasets.load_monks_dataset(i + 1)
    monks_dt = DecisionTree(MonksFeatures, name=f"Monks({i + 1}) Decision Tree")

    # We don't compute the learning curve here, cause the computation takes several minutes.
    # But you yan can compute it ny running the file: DecisionTree.py
    # For information purposes, we will still display the results of this calculation.
    # monks_dt.plot_learning_curve(monks_train, monks_train_labels,
    #                              monks_test, monks_test_labels, save_name=f"monks{i + 1}_DT", prn=prn)
    plt.title(f"Learning curve of Decision Tree - Monks({i+1})")
    plt.imshow(plt.imread(f"Figures/Learning_curve_monks{i+1}_DT.png"))
    plt.show()
    monks_dt.train(monks_train, monks_train_labels)
    cm, _, _, _ = monks_dt.test(monks_test, monks_test_labels)

    endTime = time.time() - startTime

    confusionMatrixListDT.append(cm)

    monks_dt.draw(save=False)

    print(f"\n --- Elapse time: {1_000 * endTime:.2f} ms --- \n")

    print('-' * 175)

Tpr, Fpr = util.computeTprFprList(confusionMatrixListDT, flattenOutput=False)

util.plotROCcurves(Tpr, Fpr, hmCurve=5, labels=["Iris", "Congressional", "Monks(1)", "Monks(2)", "Monks(3)"],
                   title="Decision Tree - ROC curve", save=False)


###################################################################################################################
#  Partie 2 - NeuralNet
print("##########################################################################################################")
print("                                    Partie 2 - NeuralNet                                                  ")
print("########################################################################################################## \n")
###################################################################################################################

train_ratio_nn: float = 0.7
prn = 20  # number of training per training_size for the compute of the Learning curve

confusionMatrixListNN: list = list()

print(f"NeuralNet Train ratio: {train_ratio_nn}")
print("\n")

print('-' * 175)
print(f"Iris dataset classification: \n")

startTime = time.time()

iris_train, iris_train_labels, iris_test, iris_test_labels = load_datasets.load_iris_dataset(train_ratio_nn)
nbr_output_iris = np.unique(iris_train_labels).size

# We don't compute the following stats here, cause the computation takes several minutes.
# But you yan can compute it ny running the file: NeuralNet.py
# For information purposes, we will still display the results of this calculation.
# nbr_neurons_iris = NeuralNet.get_best_number_of_hidden_neuron(iris_train, iris_train_labels, save_name="iris_nn")
plt.clf()
plt.title("Mean error by the number of hidden neurons")
plt.imshow(plt.imread("Figures/err_by_nb_neurons_iris_nn.png"))
plt.show()
nbr_neurons_iris = 6
# nbr_layer_iris = NeuralNet.get_best_number_of_hidden_layer(iris_train, iris_train_labels, iris_test, iris_test_labels,
#                                                            nbr_neurons_iris, save_name="iris")
plt.clf()
plt.title("Mean error by the number of hidden layer")
plt.imshow(plt.imread("Figures/err_by_nb_layer_iris.png"))
plt.show()
nbr_layer_iris = 1

print(f"Best number of neurones for Iris: {nbr_neurons_iris},"
      f" Best number of layers for Iris: {nbr_layer_iris}")

nn_zero_iris = NeuralNet(nbr_layer_iris, nbr_neurons_iris, nbr_output_iris, initialize_with_zeroes=True)
nn_non_zero_iris = NeuralNet(nbr_layer_iris, nbr_neurons_iris, nbr_output_iris)
# plot_RN_ZERO_RN_NON_ZERO(nn_zero_iris, nn_non_zero_iris,
#                          iris_train, iris_train_labels, iris_test, iris_test_labels, save_name="iris")
plt.clf()
plt.title("Learning curve of NeuralNet by weights initialization")
plt.imshow(plt.imread("Figures/Learning_curve_zero_vs_non_zero_iris.png"))
plt.show()

iris_nn = NeuralNet(nbr_layer_iris, nbr_neurons_iris, nbr_output_iris)
# iris_nn.plot_learning_curve(iris_train, iris_train_labels, iris_test, iris_test_labels, save_name="iris_NN",
#                             prn=prn, block=True)
plt.clf()
plt.title("Learning curve of NeuralNet - Iris")
plt.imshow(plt.imread("Figures/Learning_curve_iris_NN.png"))
plt.show()

iris_nn.train(iris_train, iris_train_labels, reset=True)
cm, _, _, _ = iris_nn.test(iris_test, iris_test_labels)

endTime = time.time() - startTime
confusionMatrixListNN.append(cm)

print(f"\n --- Elapse time: {1_000 * endTime:.2f} ms --- \n")

print('-' * 175)
print(f"Congressional dataset classification: \n")

startTime = time.time()

cong_train, cong_train_labels, cong_test, cong_test_labels = load_datasets.load_congressional_dataset(
    train_ratio_nn)
cong_train = util.replaceMissingValues(cong_train, CongressionalValue.MISSING_VALUE.value)
cong_test = util.replaceMissingValues(cong_test, CongressionalValue.MISSING_VALUE.value)
nbr_output_cong = np.unique(cong_train_labels).size

# We don't compute the following stats here, cause the computation takes several minutes.
# But you yan can compute it ny running the file: NeuralNet.py
# For information purposes, we will still display the results of this calculation.
# nbr_neurons_cong = NeuralNet.get_best_number_of_hidden_neuron(cong_train, cong_train_labels, save_name="cong_nn")
nbr_neurons_cong = 4
plt.clf()
plt.title("Mean error by the number of hidden neurons")
plt.imshow(plt.imread("Figures/err_by_nb_neurons_cong_nn.png"))
plt.show()
# nbr_layer_cong = NeuralNet.get_best_number_of_hidden_layer(cong_train, cong_train_labels, cong_test, cong_test_labels,
#                                                            nbr_neurons_cong, save_name="congressional")
nbr_layer_cong = 1
plt.clf()
plt.title("Mean error by the number of hidden layer")
plt.imshow(plt.imread("Figures/err_by_nb_layer_congressional.png"))
plt.show()

print(f"Best number of neurones for Congressional: {nbr_neurons_cong},"
      f" Best number of layers for Congressional: {nbr_layer_cong}")

nn_zero_cong = NeuralNet(nbr_layer_cong, nbr_neurons_cong, nbr_output_cong, initialize_with_zeroes=True)
nn_non_zero_cong = NeuralNet(nbr_layer_cong, nbr_neurons_cong, nbr_output_cong)

# plot_RN_ZERO_RN_NON_ZERO(nn_zero_cong, nn_non_zero_cong, cong_train, cong_train_labels, cong_test, cong_test_labels,
#                          save_name="cong")
plt.clf()
plt.title("Learning curve of NeuralNet by weights initialization")
plt.imshow(plt.imread("Figures/Learning_curve_zero_vs_non_zero_cong.png"))
plt.show()

cong_nn = NeuralNet(nbr_layer_cong, nbr_neurons_cong, nbr_output_cong)
# cong_nn.plot_learning_curve(cong_train, cong_train_labels, cong_test, cong_test_labels, save_name="cong_NN",
#                             prn=prn)
plt.clf()
plt.title("Learning curve of NeuralNet - Congressional")
plt.imshow(plt.imread("Figures/Learning_curve_cong_NN.png"))
plt.show()

cong_nn.train(cong_train, cong_train_labels, reset=True)
cm, _, _, _ = cong_nn.test(cong_test, cong_test_labels)

endTime = time.time() - startTime

confusionMatrixListNN.append(cm)

print(f"\n --- Elapse time: {1_000 * endTime:.2f} ms --- \n")

print('-' * 175)

for i in range(3):
    print(f"Monks({i + 1}) dataset classification: \n")
    startTime = time.time()

    monks_train, monks_train_labels, monks_test, monks_test_labels = load_datasets.load_monks_dataset(i + 1)
    nbr_output_monks = np.unique(monks_train_labels).size

    # We don't compute the following stats here, cause the computation takes several minutes.
    # But you yan can compute it ny running the file: NeuralNet.py
    # For information purposes, we will still display the results of this calculation.
    # nbr_neurons_monks = NeuralNet.get_best_number_of_hidden_neuron(monks_train, monks_train_labels,
    #                                                                save_name=f"monks{i + 1}_NN")
    nbr_neurons_monks = [4, 4, 7][i]
    plt.clf()
    plt.title("Mean error by the number of hidden neurons")
    plt.imshow(plt.imread(f"Figures/err_by_nb_neurons_monks{i+1}_NN.png"))
    plt.show()

    # nbr_layer_monks = NeuralNet.get_best_number_of_hidden_layer(monks_train, monks_train_labels, monks_test,
    #                                                             monks_test_labels, nbr_neurons_monks,
    #                                                             save_name=f"monks{i + 1}_NN")
    nbr_layer_monks = [1, 2, 3][i]
    plt.clf()
    plt.title("Mean error by the number of hidden layer")
    plt.imshow(plt.imread(f"Figures/err_by_nb_layer_monks{i+1}_NN.png"))
    plt.show()

    print(f"Best number of neurones for Monks({i + 1}): {nbr_neurons_monks},"
          f" Best number of layers for Monks({i + 1}): {nbr_layer_monks}")

    nn_zero_monks = NeuralNet(nbr_layer_monks, nbr_neurons_monks, nbr_output_monks, initialize_with_zeroes=True)
    nn_non_zero_monks = NeuralNet(nbr_layer_monks, nbr_neurons_monks, nbr_output_monks)
    # plot_RN_ZERO_RN_NON_ZERO(nn_zero_monks, nn_non_zero_monks, monks_train, monks_train_labels, monks_test,
    #                          monks_test_labels, save_name=f"Monks{i + 1}")
    plt.clf()
    plt.title("Learning curve of NeuralNet by weights initialization")
    plt.imshow(plt.imread(f"Figures/Learning_curve_zero_vs_non_zero_Monks{i+1}.png"))
    plt.show()

    monks_nn = NeuralNet(nbr_layer_monks, nbr_neurons_monks, nbr_output_monks)
    # monks_nn.plot_learning_curve(monks_train, monks_train_labels,
    #                              monks_test, monks_test_labels, save_name=f"monks{i + 1}_NN", prn=prn, block=True)
    plt.clf()
    plt.title(f"Learning curve of NeuralNet - Monks({i+1})")
    plt.imshow(plt.imread(f"Figures/Learning_curve_monks{i+1}_NN.png"))
    plt.show()

    monks_nn.train(monks_train, monks_train_labels, reset=True)
    cm, _, _, _ = monks_nn.test(monks_test, monks_test_labels)

    endTime = time.time() - startTime

    confusionMatrixListNN.append(cm)

    print(f"\n --- Elapse time: {1_000 * endTime:.2f} ms --- \n")

    print('-' * 175)

Tpr, Fpr = util.computeTprFprList(confusionMatrixListNN, flattenOutput=False)

util.plotROCcurves(Tpr, Fpr, hmCurve=5, labels=["Iris", "Congressional", "Monks(1)", "Monks(2)", "Monks(3)"],
                   title="Neural Net - ROC curve", save=False)

###################################################################################################################
#  Partie 3 - Comparison
print("##########################################################################################################")
print("                                    Partie 3 - Comparison                                                 ")
print("########################################################################################################## \n")
###################################################################################################################

TprDT, FprDT = util.computeTprFprList(confusionMatrixListDT)
TprNN, FprNN = util.computeTprFprList(confusionMatrixListNN)
util.plotROCcurves(np.array([TprDT, TprNN]), np.array([FprDT, FprNN]),
                   hmCurve=2, labels=["Decision Tree", "NeuralNet"], save=False)
