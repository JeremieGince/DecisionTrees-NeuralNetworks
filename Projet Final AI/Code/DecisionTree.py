from classifieur import Classifier
import numpy as np
from enum import Enum
import util
import time
from copy import deepcopy
from numpy import inf, float64, float32, int64, int32, int, float
from DecesionTreeTools import Feature, SubFeature, DISCRETE, CONTINUE, CONDITION_LABEL, Node, SubNode, Leaf, Tree


class DecisionTree(Classifier):
    def __init__(self, features: list = None, **kwargs):
        super(DecisionTree, self).__init__(**kwargs)
        self.features = deepcopy(features)
        self._entropy = None
        self._gains = None
        self.tree = None
        self.train_set = None
        self.train_labels = None
        self.name = kwargs.get("name", "Decision Tree")

    def getEntropy(self):
        return self._entropy

    @staticmethod
    def _computeEntropy(vector: np.ndarray) -> float:
        assert len(vector.shape) == 1
        labels = set(vector)
        P = [len(vector[vector == lbl])/len(vector) for lbl in labels]
        entropy = sum([-p_i*np.log2(p_i) for p_i in P])
        return entropy

    @staticmethod
    def _computeGains(vector: np.ndarray, features):
        entropy = DecisionTree._computeEntropy(vector)
        gains = np.array([(entropy - feat.getEntropy()) for feat in features])
        return gains

    @staticmethod
    def _chooseBestFeatures(data, labels, features):
        gains = DecisionTree._computeGains(labels, features)
        featuresGain: dict = {features[idx]: gains[idx] for idx in range(len(features))}
        return max(featuresGain, key=featuresGain.get)

    @staticmethod
    def _buildTree(data, labels, features, tree=None) -> Tree:
        """
        Build recursively the Decision tree to make classification based on the train data.
        :param data: The rest of the train data that remains to be classified.
        :param labels: The rest of the labels data that remains to be classified.
        :param features: The rest of Features that remains to be put in the tree.
        :param tree: The current Tree.
        :return: The Tree.
        """
        if tree is None:
            tree = Tree()

        if len(data) == 0:
            tree.addNode(Leaf(None, None), None)
            return tree
        labelsCount: dict = {lbl: len(labels[labels == lbl]) for lbl in set(labels)}
        if len(labelsCount) == 1:
            tree.addNode(Leaf(None, None, labels[0]), None)
            return tree
        if len(features) == 0:
            return tree

        best = Node(DecisionTree._chooseBestFeatures(data, labels, features))
        tree.addNode(best, None)

        data = np.delete(data, features.index(best.data), axis=1)
        features.remove(best.data)

        subFeatIdx = best.data.getRowIdxSubFeatures()
        for idx, subFeat in enumerate(best.data.subFeatures):
            if subFeat.getEntropy() > 0 and len(features) > 0:
                subNode = SubNode(subFeat, best)
                tree.addNode(subNode, subNode.parent)

                features_i = deepcopy(features)
                for jdx, feat in enumerate(features_i):
                    feat.setData(np.column_stack((data[subFeatIdx[idx], jdx], labels[subFeatIdx[idx]])))

                tree.attachTree(DecisionTree._buildTree(data[subFeatIdx[idx], :], labels[subFeatIdx[idx]],
                                                        features_i, None), subNode)
            else:
                leaf = Leaf(subFeat, best, subFeat(), subFeat.getOutLabel())
                tree.addLeaf(leaf, best)
        return tree

    def train(self, train_set: np.ndarray, train_labels: np.ndarray, verbose: bool = True, **kwargs):
        """
        :param train_set: Ensemble de données d'entraînement (np.ndarray)
        :param train_labels: Ensembles des étiquettes des données d'entraînement (np.ndarray)
        :param verbose: Vrai si nous voulons afficher certaines statistiques, Faux sinon (bool)
        :return: confusionMatrix, accuracy, precision, recall (tuple)
        """
        start_tr_time = time.time()
        if self.features is None:
            self.features = [Feature(idx) for idx in range(train_set.shape[1])]

        assert train_set.shape[1] == len(self.features),\
            f"train_set.shape[1] must be equal to {len(self.features)} but equal to {train_set.shape[1]}"
        self.train_set = train_set
        self.train_labels = train_labels

        for idx, feat in enumerate(self.features):
            feat.setData(np.column_stack((train_set[:, idx], train_labels)))

        self._entropy = self._computeEntropy(train_labels)
        self._gains = self._computeGains(train_labels, self.features)

        self.tree = self._buildTree(train_set, train_labels, deepcopy(self.features))
        isClose = self.tree.close()
        assert isClose, "Something wrong with the current tree"

        displayArgs = {"dataSize": len(train_set), "title": "Train results", "preMessage": f""}

        self.training_elapse_time = time.time() - start_tr_time
        self.prediction_elapse_times.clear()
        return self.test(train_set, train_labels, verbose, displayArgs)

    def predict(self, example, label) -> (int, bool):
        start_pr_time = time.time()
        prediction_cls = self.tree(example)
        self.prediction_elapse_times.append(time.time()-start_pr_time)
        return prediction_cls, prediction_cls == label

    def displayGains(self):
        gainsStr: str = "-"*75 + '\n'
        for idx, g in enumerate(self._gains):
            gainsStr += f"Gain(S, {self.features[idx].label}) = {g:.3f} \n"
        gainsStr += '-'*75
        print(gainsStr)

    def __str__(self):
        return str(self.tree)

    def draw(self, **kwargs):
        self.tree.draw(self.name, **kwargs)


if __name__ == '__main__':
    import load_datasets
    from load_datasets import congressionalFeatures, CongressionalValue, MonksFeatures, IrisFeatures
    import time
    import warnings

    warnings.filterwarnings("ignore")

    train_ratio_dt: float = 0.7
    prn = 20  # number of training per training_size to compute the Learning curve

    confusionMatrixList: list = list()

    print(f"Train ratio: {train_ratio_dt}")
    print("\n")

    print('-' * 175)
    print(f"Iris dataset classification: \n")
    # TODO: comparison with https://scikit-learn.org/stable/modules/tree.html#classification
    startTime = time.time()

    iris_train, iris_train_labels, iris_test, iris_test_labels = load_datasets.load_iris_dataset(train_ratio_dt)
    iris_dt = DecisionTree(IrisFeatures, name="Iris Decision Tree")
    iris_dt.plot_learning_curve(iris_train, iris_train_labels, iris_test, iris_test_labels, save_name="iris_DT", prn=prn)
    iris_dt.train(iris_train, iris_train_labels)
    cm, _, _, _ = iris_dt.test(iris_test, iris_test_labels)

    endTime = time.time() - startTime

    confusionMatrixList.append(cm)

    iris_dt.draw()

    print(f"\n --- Elapse time: {1_000 * endTime:.2f} ms --- \n")

    print('-' * 175)
    print(f"Congressional dataset classification: \n")

    startTime = time.time()

    cong_train, cong_train_labels, cong_test, cong_test_labels = load_datasets.load_congressional_dataset(train_ratio_dt)

    cong_train = util.replaceMissingValues(cong_train, CongressionalValue.MISSING_VALUE.value)
    cong_test = util.replaceMissingValues(cong_test, CongressionalValue.MISSING_VALUE.value)

    cong_dt = DecisionTree(congressionalFeatures, name="Congressional Decision Tree")
    cong_dt.plot_learning_curve(cong_train, cong_train_labels, cong_test, cong_test_labels, save_name="cong_DT", prn=prn)
    cong_dt.train(cong_train, cong_train_labels)
    cm, _, _, _ = cong_dt.test(cong_test, cong_test_labels)

    endTime = time.time() - startTime

    confusionMatrixList.append(cm)

    cong_dt.draw()

    print(f"\n --- Elapse time: {1_000 * endTime:.2f} ms --- \n")

    print('-' * 175)
    for i in range(3):
        print(f"Monks({i + 1}) dataset classification: \n")
        startTime = time.time()

        monks_train, monks_train_labels, monks_test, monks_test_labels = load_datasets.load_monks_dataset(i + 1)
        monks_dt = DecisionTree(MonksFeatures, name=f"Monks({i + 1}) Decision Tree")

        monks_dt.plot_learning_curve(monks_train, monks_train_labels,
                                     monks_test, monks_test_labels, save_name=f"monks{i + 1}_DT", prn=prn)

        monks_dt.train(monks_train, monks_train_labels)
        cm, _, _, _ = monks_dt.test(monks_test, monks_test_labels)

        endTime = time.time() - startTime

        confusionMatrixList.append(cm)

        monks_dt.draw()

        print(f"\n --- Elapse time: {1_000 * endTime:.2f} ms --- \n")

        print('-' * 175)

    Tpr, Fpr = util.computeTprFprList(confusionMatrixList, flattenOutput=False)

    util.plotROCcurves(Tpr, Fpr, hmCurve=5, labels=["Iris", "Congressional", "Monks(1)", "Monks(2)", "Monks(3)"],
                       title="Decision Tree - ROC curve")
