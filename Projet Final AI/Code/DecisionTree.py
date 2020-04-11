from classifieur import *
import numpy as np
from enum import Enum
import util
from copy import deepcopy


class FeatureType(Enum):
    DISCRETE = 0
    CONTINUE = 1


DISCRETE = FeatureType.DISCRETE
CONTINUE = FeatureType.CONTINUE


class Feature:
    def __init__(self, value: int, data: np.ndarray = None, featureType: FeatureType = DISCRETE, **kwargs):
        assert data is None or len(data.shape) == 2

        self.value = value
        self.label = kwargs.get("label", value)
        self._data = data
        self._featureType = featureType
        self._subFeatures = None

        self._entropy = None

        self._subValuesToSubFeatures: dict = dict()
        self._subValuesToSubLabels: dict = dict()
        if self._data is not None:
            self._initializeSubFeature()
            self._computeEntropy()

    def _initializeSubFeature(self):
        if self._featureType == DISCRETE:
            self._initializeSubFeatureAsDiscrete()
        elif self._featureType == CONTINUE:
            self._initializeSubFeatureAsContinue()
        else:
            raise ValueError(f"{self._featureType} is not a known type")
        self._subValuesToSubFeatures = {subFeat.value: subFeat for subFeat in self._subFeatures}
        self._subValuesToSubLabels = {subFeat.value: subFeat.label for subFeat in self._subFeatures}

    def _initializeSubFeatureAsDiscrete(self):
        self._subFeatures = [SubFeature(self, f, self._data[self._data[:, 0] == f],
                                        label=self._subValuesToSubLabels.get(f, f))
                             for idx, f in enumerate(set(self._data[:, 0]))]

    def _initializeSubFeatureAsContinue(self):
        for idx in range(self._data.shape[1]):
            pass

    @property
    def subFeatures(self):
        return self._subFeatures

    def setSubFeatures(self, newSubFeatures: list):
        assert len(newSubFeatures) == len(self._subFeatures)
        self._subFeatures = newSubFeatures
        for subFeat in self._subFeatures:
            subFeat.parent = self
        self._subValuesToSubFeatures = {subFeat.value: subFeat for subFeat in self._subFeatures}
        self._subValuesToSubLabels = {subFeat.value: subFeat.label for subFeat in self._subFeatures}

    def getEntropy(self) -> float:
        return self._entropy

    def setData(self, newData):
        assert len(newData.shape) == 2
        self._data = newData
        self._initializeSubFeature()
        self._computeEntropy()

    def getData(self):
        return self._data

    def __len__(self):
        return len(self._data)

    def _computeEntropy(self):
        H = [subFeat.getEntropy() for subFeat in self._subFeatures]
        W = [len(subFeat) / len(self) for subFeat in self._subFeatures]
        self._entropy = sum([W[i] * H[i] for i in range(len(self._subFeatures))])
        return self._entropy

    def getRowIdxSubFeatures(self):
        return [np.where(self._data[:, 0] == subFeat.value)[0] for subFeat in self._subFeatures]

    def displayEntropy(self):
        this: str = "-"*75 + '\n'
        this += f"Entropy({self.label}) = {self._entropy:.3f}"
        for subFeat in self._subFeatures:
            this += f"\n\t{str(subFeat)}"
        this += '\n'+"-"*75
        return this

    def __str__(self):
        return f"{str(self.label)}"+'{'+f"{', '.join([str(subFeat.label) for subFeat in self.subFeatures])}"+'}'

    def __call__(self, vector: np.ndarray):
        return self._subValuesToSubFeatures.get(vector[self.value])


class SubFeature(Feature):
    def __init__(self, parent: Feature, value: int, data: np.ndarray = None,
                 featureType: FeatureType = DISCRETE, **kwargs):
        assert data is None or len(data.shape) == 2

        self.value = value
        self.label = kwargs.get("label", value)
        self._data = data
        self._featureType = featureType
        self._entropy = None

        if data is not None:
            self._computeEntropy()
        self.parent = parent

    def _computeEntropy(self):
        labels = set(self._data[:, -1])
        P = [len(self._data[self._data[:, -1] == lbl]) / len(self._data) for lbl in labels]
        self._entropy = sum([-p_i * np.log2(p_i) for p_i in P])
        return self._entropy

    def displayEntropy(self):
        return f"Entropy({self.parent.label}.{self.label}) = {self._entropy:.3f}"

    def __str__(self):
        return f"{self.parent.label}.{self.label}"

    def __call__(self, vector: np.ndarray = None):
        labels = set(self._data[:, -1])
        labelsCount: dict = {lbl: len(self._data[:, -1][self._data[:, -1] == lbl]) for lbl in set(labels)}
        return max(labelsCount, key=labelsCount.get)


class Branch:
    def __init__(self, parent, child, label, data=None):
        self.parent = parent
        self.child = child
        self.label = label
        self.data = data


class Node:
    def __init__(self, data: Feature, parent=None, children: list = None, **kwargs):
        self._data = data
        self._parent = parent
        self._children = list() if children is None else children
        self.labels = kwargs.get("labels", list(range(len(self._children))))
        assert len(self.labels) == len(self._children)
        self.branches = [Branch(parent, child, self.labels[idx]) for idx, child in enumerate(self._children)]
        self.subNodeValuesToSubNodes = {subNode.data.value: subNode for subNode in self._children}
        self.subFeatureToSubNode = {subNode.data: subNode for subNode in self._children}
        self.isClose = False

    @property
    def data(self):
        return self._data

    @property
    def parent(self):
        return self._parent

    def setData(self, newData):
        self._data = newData

    def setParent(self, newParent):
        self._parent = newParent

    def setChildren(self, newChildren, labels=None):
        self._children = newChildren
        if labels is None:
            self.labels = list(range(len(newChildren)))
        else:
            self.labels = labels
        self.branches = [Branch(self._parent, child, self.labels[idx]) for idx, child in enumerate(newChildren)]

    def addChild(self, newChild, label=None):
        self._children.append(newChild)
        if label is None:
            label = len(self.labels)
        self.labels.append(label)
        self.branches.append(Branch(self._parent, newChild, label))

    @property
    def children(self):
        return self._children

    def __str__(self):
        return f"{self.data}"

    def __call__(self, vector: np.ndarray):
        return self.subFeatureToSubNode[self._data(vector)](vector)

    def close(self):
        assert len(self._children) > 0
        assert self._parent is None or isinstance(self._parent, SubNode)
        self.subNodeValuesToSubNodes = {subNode.data.value: subNode for subNode in self._children}
        self.subFeatureToSubNode = {subNode.data: subNode for subNode in self._children}
        if self._parent is not None:
            self._parent.close()
        self.isClose = True


class SubNode(Node):
    def __init__(self, data: Feature, parent: Node):
        super(SubNode, self).__init__(data, parent, None)

    def __call__(self, vector: np.ndarray):
        return self._children[0](vector)

    def close(self):
        assert len(self._children) == 1
        assert self._parent is not None
        assert isinstance(self._parent, Node)

        self.subNodeValuesToSubNodes = {subNode.data.value: subNode for subNode in self._children}
        self._parent.close()
        self.isClose = True


class Leaf(Node):
    def __init__(self, data: Feature, parent: Node, info=None):
        super(Leaf, self).__init__(data, parent, None)
        self.info = info

    def setChildren(self, newChildren, labels=None):
        raise NotImplementedError()

    def __call__(self, vector: np.ndarray = None):
        return self.info

    def close(self):
        assert len(self._children) == 0
        assert self._parent is not None
        assert isinstance(self._parent, (SubNode, Node))

        if self.data is None:
            self._data = self._parent.data
        if self.info is None:
            if isinstance(self._parent.data, Feature):
                self.info = self._parent.data()()
            if isinstance(self._parent.data, SubFeature):
                self.info = self._parent.data()
        self._parent.close()
        self.isClose = True

    def __str__(self):
        return f"Leaf: {self.data} -> {self()}"


class Tree:
    def __init__(self, root: Node = None):
        self.root = root
        self.nodes: list = list()
        self.leaves = list()
        self.isClose = False

    def setRoot(self, newRoot: Node):
        self.root = newRoot

    def addNode(self, node: Node, parent: Node):
        if isinstance(node, Leaf):
            self.leaves.append(node)

        self.nodes.append(node)
        if self.root is None:
            if parent is not None:
                self.setRoot(parent)
            else:
                self.setRoot(node)
        node.setParent(parent)
        if parent is not None:
            parent.addChild(node)

    def attachTree(self, other, parent: Node):
        if other is None or other.root is None:
            return
        self.addNode(other.root, parent)
        for oth_n in other.nodes:
            self.nodes.append(oth_n)
        for oth_l in other.leaves:
            self.leaves.append(oth_l)

    def addLeaf(self, node: Leaf, parent: Node):
        self.addNode(node, parent)

    def __call__(self, vector: np.ndarray):
        assert self.isClose
        return self.root(vector)

    def close(self):
        for leaf in self.leaves:
            leaf.close()
        self.isClose = all([node.isClose for node in self.nodes])
        return self.isClose

    def __str__(self):
        if self.root is None:
            return None
        this = ""
        childrenList = [self.root.children]
        this += "Root -> "
        this += str(self.root) + '\n'

        layer = 1
        while childrenList:
            this += '-'*50 + '\n'
            this += f"Layer {layer} -> "
            for children in childrenList:
                if children:
                    this += str([str(c).replace('-', '') for c in children]) + ', '
            this += '\n'

            nextChildrenList = []
            for a in childrenList:
                for c in a:
                    nextChildrenList.append(c.children)

            childrenList = nextChildrenList
            layer += 1
        this += '-' * 50 + '\n'
        return this


class DecisionTree(Classifier):
    def __init__(self, features: list = None, **kwargs):
        super(DecisionTree, self).__init__(**kwargs)
        self.features = features
        self._entropy = None
        self._gains = None
        self.tree = None
        self.train_set = None
        self.train_labels = None

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
            # return max(labelsCount, key=labelsCount.get)
            return tree

        best = Node(DecisionTree._chooseBestFeatures(data, labels, features))
        tree.addNode(best, None)
        # m = max(labelsCount, key=labelsCount.get)

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
                leaf = Leaf(subFeat, best, subFeat())
                tree.addLeaf(leaf, best)
        return tree

    def train(self, train_set: np.ndarray, train_labels: np.ndarray, verbose: bool = True, **kwargs):
        """
        :param train_set: Ensemble de données d'entraînement (np.ndarray)
        :param train_labels: Ensembles des étiquettes des données d'entraînement (np.ndarray)
        :param verbose: Vrai si nous voulons afficher certaines statistiques, Faux sinon (bool)
        :return: confusionMatrix, accuracy, precision, recall (tuple)
        """
        if self.features is None:
            self.features = [Feature(idx) for idx in range(train_set.shape[1])]

        assert train_set.shape[1] == len(self.features)
        self.train_set = train_set
        self.train_labels = train_labels

        for idx, feat in enumerate(self.features):
            feat.setData(np.column_stack((train_set[:, idx], train_labels)))

        self._entropy = self._computeEntropy(train_labels)
        self._gains = self._computeGains(train_labels, self.features)

        self.tree = self._buildTree(train_set, train_labels, self.features)
        isClose = self.tree.close()
        assert isClose, "Something wrong with the current tree"

        displayArgs = {"dataSize": len(train_set), "title": "Train results", "preMessage": f" \n"}

        return self.test(train_set, train_labels, verbose, displayArgs)

    def predict(self, example, label) -> (int, bool):
        prediction_cls = self.tree(example)
        return prediction_cls, prediction_cls == label

    def displayGains(self):
        gainsStr: str = "-"*75 + '\n'
        for idx, g in enumerate(self._gains):
            gainsStr += f"Gain(S, {self.features[idx].label}) = {g:.3f} \n"
        gainsStr += '-'*75
        print(gainsStr)

    def __str__(self):
        return str(self.tree)


if __name__ == '__main__':
    import load_datasets
    import time

    train_ratio_dt: float = 0.8

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