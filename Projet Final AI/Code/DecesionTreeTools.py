from classifieur import Classifier
import numpy as np
from enum import Enum
import util
from copy import deepcopy
from numpy import inf, float64, float32, int64, int32, int, float


class FeatureType(Enum):
    DISCRETE = 0
    CONTINUE = 1


DISCRETE = FeatureType.DISCRETE
CONTINUE = FeatureType.CONTINUE

CONDITION_LABEL = '$x_i$'


class Feature:
    """
    Class Feature used to manipulate the attributes in a dataset.
    """
    def __init__(self, value: int, data: np.ndarray = None, featureType: FeatureType = None, **kwargs):
        """
        Constructor of Feature.
        :param value: The value of the feature
                      (i.e. the index of the attribute in an exemple vector of the current dataset) (:type: int)
        :param data: The data of the feature. A numpy array [attribute values, labels]. (:type: np.ndarray)
        :param featureType: The type of the feature, DISCRETE or CONTINUE. (:type: FeatureType)
        :param kwargs:
                        label: The name of the feature, default: value (:type: str)
        """
        assert data is None or len(data.shape) == 2

        self.value = value
        self.label = kwargs.get("label", value)
        self.outLabels = kwargs.get("outLabels", dict())
        self._data = data
        self._featureType = featureType
        self._subFeatures = None

        self._entropy = None

        self._subValuesToSubFeatures: dict = dict()
        self._subValuesToSubLabels: dict = dict()
        if self._data is not None:
            self._initializeSubFeature()
            self._computeEntropy()

    def _detectFeatureType(self):
        """
        Detect if the current feature is Discrete or Continue,
        but it's recommended to specified it in the constructor.
        :return: None
        """
        if self._data.dtype in [int, np.int, np.int32, np.int64, bool, np.bool] \
                and len(set(self._data[:, 0])) < len(self._data)/2:
            self._featureType = DISCRETE
        else:
            self._featureType = CONTINUE

    def _initializeSubFeature(self):
        if self._featureType == DISCRETE:
            self._initializeSubFeatureAsDiscrete()
        elif self._featureType == CONTINUE:
            self._initializeSubFeatureAsContinue()
        else:
            self._detectFeatureType()
            self._initializeSubFeature()
        self._subValuesToSubFeatures = {subFeat.value: subFeat for subFeat in self._subFeatures}
        self._subValuesToSubLabels = {subFeat.value: subFeat.label for subFeat in self._subFeatures}

    def _initializeSubFeatureAsDiscrete(self):
        self._subFeatures = [SubFeature(self, f, self._data[self._data[:, 0] == f],
                                        label=self._subValuesToSubLabels.get(f, f))
                             for idx, f in enumerate(set(self._data[:, 0]))]

    def _initializeSubFeatureAsContinue(self):
        """
        Si le type de feature est continue, alors

        Considérant k le nombre de classes c de l'ensemble de données, pour chacune de ces classes c_i,
            1. trouver les intervales de valeurs du feature courant correspondant à chaque classes.
            2. Créer k nouvelles branches ayant comme condition: "être à l'intérieur des
               intervales de valeurs associés à la classe c_i".
            3. Élaguer les branches non nécessaires. (À faire)

        De cette façon, si la fonction gouvernant les valeurs du fearture courant est non linéaire, alors les intervales
        associés aux classes représenteront cette tandence, ce qui était un problème avec les points de coupure, car
        ceux-ci  approximait la fonction comme linéaire et généraient une grande perte d'information. De plus, si une ou
        plusieurs branches sont créées inutilement, elles seront élagué plus tard. L'avantage des intervales de
        valeurs sur les points de  coupure est que les intervales de valeurs vont approximer la fonction un peu
        comme une somme de reimann le ferait, tandis que le point de coupure va approximer la fonction comme linéaire
        ce qui n'est absolument pas valide pour une fonction oscillante comme un sinus.

        :return: None
        """
        self._subFeatures: list = list()
        dataSorted = np.array(self._data[self._data[:, 0].argsort()])

        lblToIntervals: dict = {lbl: [] for lbl in set(self._data[:, -1])}

        currLbl = dataSorted[0, -1]
        currVal = -np.inf
        for j in range(len(dataSorted[:, 0])):
            if currLbl != dataSorted[j, -1]:
                lblToIntervals[currLbl].append([currVal, dataSorted[j, 0]])
                currLbl = dataSorted[j, -1]
                currVal = dataSorted[j, 0]

        lblToIntervals[currLbl].append([currVal, np.inf])

        for lbl, listOfInterval in lblToIntervals.items():
            reducedData = self._data[np.any(np.array([((low <= self._data[:, 0]) * (self._data[:, 0] < upp))
                                                      for low, upp in listOfInterval]), axis=0)]
            if reducedData.size == 0:
                # TODO: essayer de s'arranger pour faire en sorte que la premiere classe ne prenne pas tout
                #       le domaine des réels pour en laisser aux suivantes
                continue
            condition = f"np.any(np.array([((low <= {CONDITION_LABEL}) * ({CONDITION_LABEL} < upp)) " \
                        f"\n for low, upp in {listOfInterval}]), axis=0)"
            self._subFeatures.append(SubFeature(self, lbl,
                                                reducedData,
                                                label=self._subValuesToSubLabels.get(lbl, lbl),
                                                condition=condition))

    @property
    def subFeatures(self):
        return self._subFeatures

    def setSubFeatures(self, newSubFeatures: list):
        """
        Setter of the subfeatures of the current feature.
         It's commonly the values that the current feature can take in the dataset.
        :param newSubFeatures: The subFeature. :type: list[SubFeature]
        :return: None
        """
        self._subFeatures = newSubFeatures
        for subFeat in self._subFeatures:
            assert isinstance(subFeat, SubFeature), "newSubFeatures must be a list of SubFeature"
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
        """
        Compute the Shannon entropy of the current feature.

            $entropy(S) = \sum_{\nu \in V} \frac{\norm{S_\nu}}{\norm{S}} * entropy(S_{\nu})$

        :return: the Shannon entropy.
        """
        H = [subFeat.getEntropy() for subFeat in self._subFeatures]
        W = [len(subFeat) / len(self) for subFeat in self._subFeatures]
        self._entropy = sum([W[i] * H[i] for i in range(len(self._subFeatures))])
        return self._entropy

    def getRowIdxSubFeatures(self):
        return [np.where(subFeat.evalCondition(self._data[:, 0]))[0] for subFeat in self._subFeatures]

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
        """
        Return the subFeature that respect it's condition with the input vector.
        :param vector: vector, is a sample of the dataset. (:type: np.ndarray)
        :return: The subFeature that is link with the input vector.
        """
        if self._featureType == DISCRETE:
            return self._subValuesToSubFeatures.get(vector[self.value],
                                                    np.random.choice(list(self._subValuesToSubFeatures.values())))
        elif self._featureType == CONTINUE:
            reSubFeat = self._subFeatures[0]
            for subFeat in self._subFeatures:
                if subFeat.evalCondition(vector[self.value]):
                    reSubFeat = subFeat
                    break
            return reSubFeat
        else:
            raise ValueError()


class SubFeature(Feature):
    """
    Class SubFeature used to interpret the possible values for it's feature in the dataset.
    """
    def __init__(self, parent: Feature, value: int, data: np.ndarray = None, **kwargs):
        """
        Contructor of SubFeature.

        :param parent: The parent Feature. (:type: Feature)
        :param value: The value of the subFeature. (:type: int)
        :param data: The data corresponding og the subFeature.
        :param kwargs:
                        :param condition: the condition that the current SubFeature has to respect.
        """
        assert data is None or len(data.shape) == 2

        self.value = value
        self.condition = kwargs.get("condition", f"{CONDITION_LABEL} == {value}")
        self.label = kwargs.get("label", value)
        self._data = data
        self._featureType = DISCRETE
        self._entropy = None

        if data is not None:
            self._computeEntropy()
        self.parent = parent

    def _computeEntropy(self) -> float:
        """
        Compute the Shannon entropy for the current SubFeature.

            $entropy(S_{\nu}) = \sum_i -p_i \log_2{(p_i)}$

        :return: The Shannon entropy. :rtype: float
        """
        labels = set(self._data[:, -1])
        P = [len(self._data[self._data[:, -1] == lbl]) / len(self._data) for lbl in labels]
        self._entropy = sum([-p_i * np.log2(p_i) for p_i in P])
        return self._entropy

    def displayEntropy(self) -> str:
        return f"Entropy({self.parent.label}.{self.label}) = {self._entropy:.3f}"

    def __str__(self) -> str:
        # f"{self.parent.label}.{self.label} \n {self.condition}"
        return f"{self.parent.label}.{self.label}"

    def __call__(self, vector: np.ndarray = None) -> int:
        """
        Get the label (i.e. the cls for the classification) associated with the current SubFeature.
        :param vector: The input vector that will be classify.
        :return: The classification of the input vector. :rtype: int
        """
        labels = set(self._data[:, -1])
        labelsCount: dict = {lbl: len(self._data[:, -1][self._data[:, -1] == lbl]) for lbl in set(labels)}
        return max(labelsCount, key=labelsCount.get)

    def getOutLabel(self) -> str:
        outValue = self()
        return str(self.parent.outLabels.get(outValue, outValue))

    def evalCondition(self, inputValue) -> bool:
        """
        Evaluate if the input vector respect the current condition to be in the SubFeature.
        :param inputValue: Input vector :type: np.nadarray or float or int
        :return: True if the input respect the condition else False. :type: bool
        """
        if isinstance(inputValue, np.ndarray):
            return eval(self.condition.replace(CONDITION_LABEL,
                                               f"np.fromstring({inputValue.tostring()},"
                                               f"{inputValue.dtype}).reshape({inputValue.shape})"))
        return eval(self.condition.replace(CONDITION_LABEL, str(inputValue)))


class Branch:
    """
    class Branch used to link two nodes in a graph.
    """
    def __init__(self, parent, child, label, data=None):
        self.parent = parent
        self.child = child
        self.label = label
        self.data = data


class Node:
    """
    Class Node used to contain a Feature in a tree graph.
    """
    def __init__(self, data: Feature, parent=None, children: list = None, **kwargs):
        """
        Constructor of Node.
        :param data: The data of the current Node. Must be a Feature. :type: Feature.
        :param parent: The parent of the current Node. Must be a SubFeature or None
                        if the current Node is the root. :type: SubFeature
        :param children: List of the children of the current Node: All child must be a SubFeature.
                         :type: list[SubFeature]
        :param kwargs:
                        :param labels: Labels to represent the current children of the Node.
        """
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
        """
        Will call it's current data with the input vector.
        :param vector: input vector. :type: np.ndarray
        :return: The output of the current data.
        """
        return self.subFeatureToSubNode[self._data(vector)](vector)

    def close(self):
        """
        Make sur that the current node is ready to be called.
        :return: None
        """
        assert len(self._children) > 0
        assert self._parent is None or isinstance(self._parent, SubNode)
        self.subNodeValuesToSubNodes = {subNode.data.value: subNode for subNode in self._children}
        self.subFeatureToSubNode = {subNode.data: subNode for subNode in self._children}
        if self._parent is not None:
            self._parent.close()
        self.isClose = True


class SubNode(Node):
    """
    Class SubNode used to contain a SubFeature in a tree graph.
    """
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
    """
        Class SubNode used to contain a SubFeature in a tree graph as a leaf node.
    """
    def __init__(self, data: Feature, parent: Node, out=None, info=None):
        """
        Constructor of Leaf.
        :param data: The data of the current Node. Must be a Feature. :type: Feature.
        :param parent: The parent of the current Node. Must be a Feature or None
                        if the current Node is the root. :type: Feature
        :param out: The information has be returned when call.
        """
        super(Leaf, self).__init__(data, parent, None)
        self.out = out
        self.info = str(out) if info is None else info

    def setChildren(self, newChildren, labels=None):
        raise NotImplementedError()

    def __call__(self, vector: np.ndarray = None):
        return self.out

    def close(self):
        assert len(self._children) == 0
        assert self._parent is not None
        assert isinstance(self._parent, (SubNode, Node))

        if self.data is None:
            self._data = self._parent.data
        if self.out is None:
            if isinstance(self._parent.data, Feature):
                self.out = self._parent.data()()
                self.info = self._parent.data().getOutLabel()
            if isinstance(self._parent.data, SubFeature):
                self.out = self._parent.data()
                self.info = self._parent.data.getOutLabel()
        self._parent.close()
        self.isClose = True

    def __str__(self):
        return f"{self.data} \n out: {self.info}"


class Tree:
    """
    class Tree used to manipulate and contain Nodes in a tree graph.
    """
    def __init__(self, root: Node = None):
        """
        Constructo of Tree.
        :param root: The root Node. :type: Node
        """
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
        """
        Will call the root node and the root node will call every subNodes recursivly.
        :param vector: The input vector to pass to root.
        :return: The output of the current Tree.
        """
        assert self.isClose
        return self.root(vector)

    def close(self):
        """
        Make sur that the current Tree is ready to be called.
        :return: True if ready else False :rtype: bool
        """
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

    def getNodesAsTuples(self):
        return [(node.parent, node) for node in self.nodes]

    def draw(self, title=f"Tree"):
        from DrawingTreeGraph import drawTree
        drawTree(self, title=title)
