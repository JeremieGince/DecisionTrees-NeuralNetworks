import load_datasets
from load_datasets import congressionalFeatures, CongressionalValue, MonksFeatures, IrisFeatures
import time
import warnings
from DecisionTree import DecisionTree
import util
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore")


###################################################################################################################
#  Partie 1 - Decision Tree
###################################################################################################################


train_ratio_dt: float = 0.7
prn = 20  # number of training per training_size to compute the Learning curve

confusionMatrixListDT: list = list()

print(f"Train ratio: {train_ratio_dt}")
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
# cong_dt.plot_learning_curve(cong_train, cong_train_labels, cong_test, cong_test_labels, save_name="cong_DT", prn=prn)
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
###################################################################################################################

confusionMatrixListNN: list = list()

###################################################################################################################
#  Partie 3 - Comparison
###################################################################################################################

TprDT, FprDT = util.computeTprFprList(confusionMatrixListDT)
TprNN, FprNN = util.computeTprFprList(confusionMatrixListNN)
util.plotROCcurves(np.array([TprDT, TprNN]), np.array([FprDT, FprNN]),
                   hmCurve=2, labels=["Decision Tree", "NeuralNet"])
