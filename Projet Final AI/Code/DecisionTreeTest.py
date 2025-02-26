from DecisionTree import DecisionTree, Feature, DISCRETE, CONTINUE, SubFeature
from enum import Enum
import numpy as np
import time
import util
from DrawingTreeGraph import drawTree
import warnings

warnings.filterwarnings("ignore")

# Tennis exemple


class TennisFeatures(Enum):
    ENSOLEILLE = 0
    NUAGEUX = 1
    PLUVIEUX = 2
    CHAUDE = 3
    TEMPEREE = 4
    FROIDE = 5
    ELEVEE = 6
    NORMAL = 7
    FAIBLE = 8
    FORT = 9
    NON = False
    OUI = True


ENSOLEILLE = TennisFeatures.ENSOLEILLE
NUAGEUX = TennisFeatures.NUAGEUX
PLUVIEUX = TennisFeatures.PLUVIEUX
CHAUDE = TennisFeatures.CHAUDE
TEMPEREE = TennisFeatures.TEMPEREE
FROIDE = TennisFeatures.FROIDE
ELEVEE = TennisFeatures.ELEVEE
NORMAL = TennisFeatures.NORMAL
FAIBLE = TennisFeatures.FAIBLE
FORT = TennisFeatures.FORT
NON = TennisFeatures.NON
OUI = TennisFeatures.OUI


if __name__ == '__main__':

    data = [
        [ENSOLEILLE, CHAUDE, ELEVEE, FAIBLE, NON],
        [ENSOLEILLE, CHAUDE, ELEVEE, FORT, NON],
        [NUAGEUX, CHAUDE, ELEVEE, FAIBLE, OUI],
        [PLUVIEUX, TEMPEREE, ELEVEE, FAIBLE, OUI],
        [PLUVIEUX, FROIDE, NORMAL, FAIBLE, OUI],
        [PLUVIEUX, FROIDE, NORMAL, FORT, NON],
        [NUAGEUX, FROIDE, NORMAL, FORT, OUI],
        [ENSOLEILLE, TEMPEREE, ELEVEE, FAIBLE, NON],
        [ENSOLEILLE, FROIDE, NORMAL, FAIBLE, OUI],
        [PLUVIEUX, TEMPEREE, NORMAL, FAIBLE, OUI],
        [ENSOLEILLE, TEMPEREE, NORMAL, FORT, OUI],
        [NUAGEUX, TEMPEREE, ELEVEE, FORT, OUI],
        [NUAGEUX, CHAUDE, NORMAL, FAIBLE, OUI],
        [PLUVIEUX, TEMPEREE, ELEVEE, FORT, NON]
    ]

    tempData = []
    for vec in data:
        line = []
        for en in vec:
            line.append(en.value)
        tempData.append(line)
    data = np.array(tempData, dtype=int)

    cielFeature = Feature(0, data[:, [0, -1]], featureType=DISCRETE, label="ciel",
                          outLabels={OUI.value: "Oui", NON.value: "Non"})
    cielFeature.setSubFeatures([
        SubFeature(cielFeature, ENSOLEILLE.value, data[:, [0, -1]][data[:, 0] == ENSOLEILLE.value], label="Ensoleillé"),
        SubFeature(cielFeature, NUAGEUX.value, data[:, [0, -1]][data[:, 0] == NUAGEUX.value], label="Nuageux"),
        SubFeature(cielFeature, PLUVIEUX.value, data[:, [0, -1]][data[:, 0] == PLUVIEUX.value], label="Pluvieux")
    ])

    temperatureFeature = Feature(1, data[:, [1, -1]], featureType=DISCRETE, label="Température",
                          outLabels={OUI.value: "Oui", NON.value: "Non"})
    temperatureFeature.setSubFeatures([
        SubFeature(temperatureFeature, CHAUDE.value, data[:, [1, -1]][data[:, 1] == CHAUDE.value], label="Chaude"),
        SubFeature(temperatureFeature, TEMPEREE.value, data[:, [1, -1]][data[:, 1] == TEMPEREE.value], label="Tempérée"),
        SubFeature(temperatureFeature, FROIDE.value, data[:, [1, -1]][data[:, 1] == FROIDE.value], label="Froide"),
    ])

    humiditeFeature = Feature(2, data[:, [2, -1]], featureType=DISCRETE, label="Humidité",
                          outLabels={OUI.value: "Oui", NON.value: "Non"})
    humiditeFeature.setSubFeatures([
        SubFeature(humiditeFeature, ELEVEE.value, data[:, [2, -1]][data[:, 2] == ELEVEE.value], label="Élevée"),
        SubFeature(humiditeFeature, NORMAL.value, data[:, [2, -1]][data[:, 2] == NORMAL.value], label="Normal"),
    ])

    ventFeature = Feature(3, data[:, [3, -1]], featureType=DISCRETE, label="Vent",
                          outLabels={OUI.value: "Oui", NON.value: "Non"})
    ventFeature.setSubFeatures([
        SubFeature(ventFeature, FAIBLE.value, data[:, [3, -1]][data[:, 3] == FAIBLE.value], label="Faible"),
        SubFeature(ventFeature, FORT.value, data[:, [3, -1]][data[:, 3] == FORT.value], label="Fort"),
    ])

    features = [cielFeature, temperatureFeature, humiditeFeature, ventFeature]

    confusionMatrixList: list = list()

    print("\n")

    startTime = time.time()

    for feat in features:
        print(feat.displayEntropy())

    dtree = DecisionTree(features, name="Decision Tree - Test Tennis")


    # dtree._computeGains(data[:, -1])
    # dtree.displayGains()
    dtree.train(data[:, :-1], data[:, -1])
    print(f"Entropy(S) = {dtree._computeEntropy(data[:, -1])}")

    print('\n The Tree: \n' + str(dtree))

    cm, _, _, _ = dtree.test(data[:, :-1], data[:, -1])
    confusionMatrixList.append(cm)

    Tpr, Fpr = util.computeTprFprList(confusionMatrixList, flattenOutput=False)

    util.plotROCcurves(Tpr, Fpr, hmCurve=1, labels=["Tennis"], title="Decision Tree - Test Tennis")

    print(f"\n --- Elapse time: {1_000 * (time.time() - startTime):.2f} ms --- \n")

    dtree.draw()
