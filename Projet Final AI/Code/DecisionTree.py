from classifieur import *
import numpy as np
import util


class DecisionTree(Classifier):
    def train(self, train_set: np.ndarray, train_labels: np.ndarray, verbose: bool = True, **kwargs):
        pass

    def predict(self, example, label) -> (int, bool):
        pass


if __name__ == '__main__':
    import load_datasets
    import time

    train_ratio_dt: float = 0.9

    confusionMatrixList: list = list()

    print(f"Train ratio: {train_ratio_dt}")
    print("\n")

    print('-' * 175)
    print(f"Iris dataset classification: \n")
    startTime = time.time()

    iris_train, iris_train_labels, iris_test, iris_test_labels = load_datasets.load_iris_dataset(train_ratio_dt)
    iris_dt = DecisionTree()
    iris_dt.train(iris_train, iris_train_labels)
    cm, _, _, _ = iris_dt.test(iris_test, iris_test_labels)
    confusionMatrixList.append(cm)

    print(f"\n --- Elapse time: {1_000 * (time.time() - startTime):.2f} ms --- \n")

    print('-' * 175)
    print(f"Congressional dataset classification: \n")
    startTime = time.time()

    cong_train, cong_train_labels, cong_test, cong_test_labels = load_datasets.load_congressional_dataset(train_ratio_dt)
    cong_dt = DecisionTree()
    cong_dt.train(cong_train, cong_train_labels)
    cm, _, _, _ = cong_dt.test(cong_test, cong_test_labels)
    confusionMatrixList.append(cm)

    print(f"\n --- Elapse time: {1_000 * (time.time() - startTime):.2f} ms --- \n")

    print('-' * 175)
    for i in range(3):
        print(f"Monks({i + 1}) dataset classification: \n")
        startTime = time.time()

        monks_train, monks_train_labels, monks_test, monks_test_labels = load_datasets.load_monks_dataset(i + 1)
        monks_dt = DecisionTree()
        monks_dt.train(monks_train, monks_train_labels)
        cm, _, _, _ = monks_dt.test(monks_test, monks_test_labels)
        confusionMatrixList.append(cm)

        print(f"\n --- Elapse time: {1_000 * (time.time() - startTime):.2f} ms --- \n")

        print('-' * 175)

    Tpr, Fpr = util.computeTprFprList(confusionMatrixList)

    print(Tpr, Fpr, sep='\n')

    util.plotROCcurves(Tpr, Fpr, labels=["Decision Tree"])