import random
from enum import Enum

import numpy as np

from DecesionTreeTools import Feature, SubFeature, DISCRETE, CONTINUE

sepalLengthFeature = Feature(0, featureType=CONTINUE, label="sepal length [cm]",
                             outLabels={0: 'Iris-setosa', 1: 'Iris-versicolor', 2: 'Iris-virginica'})
sepalWidthFeature = Feature(1, featureType=CONTINUE, label="sepal width [cm]",
                            outLabels={0: 'Iris-setosa', 1: 'Iris-versicolor', 2: 'Iris-virginica'})
petalLengthFeature = Feature(2, featureType=CONTINUE, label="petal length [cm]",
                             outLabels={0: 'Iris-setosa', 1: 'Iris-versicolor', 2: 'Iris-virginica'})
petalWidthFeature = Feature(3, featureType=CONTINUE, label="petal width [cm]",
                            outLabels={0: 'Iris-setosa', 1: 'Iris-versicolor', 2: 'Iris-virginica'})

IrisFeatures = [
    sepalLengthFeature, sepalWidthFeature, petalLengthFeature, petalWidthFeature
]


def load_iris_dataset(train_ratio: float) -> tuple:
    """Cette fonction a pour but de lire le dataset Iris

    datset reference: http://archive.ics.uci.edu/ml/datasets/iris

    Args:
        train_ratio: le ratio des exemples (ou instances) qui vont etre attribués à l'entrainement,
        le rest des exemples va etre utilisé pour les tests.
        Par example : si le ratio est 50%, il y aura 50% des example (75 exemples) qui vont etre utilisé
        pour l'entrainement, et 50% (75 exemples) pour le test_set.

    Retours:
        Cette fonction doit retourner 4 matrices de type Numpy, train_set, train_labels, test_set, et test_labels

        - train_set : une matrice numpy qui contient les exemples qui vont etre utilisés pour l'entrainement, chaque
        ligne dans cette matrice représente un example (ou instance) d'entrainement.

        - train_labels : contient les labels (ou les étiquettes) pour chaque example dans train_set, de telle sorte
          que : train_labels[i] est le label (ou l'etiquette) pour l'example train_set[i]

        - test_set : une matrice numpy qui contient les exemples qui vont etre utilisés pour le test_set, chaque
        ligne dans cette matrice représente un example (ou instance) de test_set.

        - test_labels : contient les labels (ou les étiquettes) pour chaque example dans test_set, de telle sorte
          que : test_labels[i] est le label (ou l'etiquette) pour l'example test_set[i]
    """

    random.seed(1)  # Pour avoir les meme nombres aléatoires à chaque initialisation.

    # Vous pouvez utiliser des valeurs numériques pour les différents types de classes, tel que :
    conversion_labels = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}

    data, data_labels = extract_raw_data('datasets/bezdekIris.data',
                                         class_index=-1, conversion_labels=conversion_labels)

    split_idx: int = int(len(data) * train_ratio)

    train: np.ndarray = np.array(data[:split_idx])
    train_labels: np.ndarray = np.array(data_labels[:split_idx])

    test: np.ndarray = np.array(data[split_idx:])
    test_labels: np.ndarray = np.array(data_labels[split_idx:])

    return train, train_labels, test, test_labels


class CongressionalValue(Enum):
    N = 0
    Y = 1
    MISSING_VALUE = 2


handicappedInfantsFeature = Feature(0, featureType=DISCRETE, label="handicapped-infants",
                                    outLabels={0: 'republican', 1: 'democrat'})
handicappedInfantsFeature.setSubFeatures([
    SubFeature(handicappedInfantsFeature, CongressionalValue.N.value, label="n"),
    SubFeature(handicappedInfantsFeature, CongressionalValue.Y.value, label="y"),
])

waterProjectCostSharingFeature = Feature(1, featureType=DISCRETE, label="water-project-cost-sharing",
                                         outLabels={0: 'republican', 1: 'democrat'})
waterProjectCostSharingFeature.setSubFeatures([
    SubFeature(waterProjectCostSharingFeature, CongressionalValue.N.value, label="n"),
    SubFeature(waterProjectCostSharingFeature, CongressionalValue.Y.value, label="y"),
])

adoptionOfTheBudgetResolutionFeature = Feature(2, featureType=DISCRETE, label="adoption-of-the-budget-resolution",
                                               outLabels={0: 'republican', 1: 'democrat'})
adoptionOfTheBudgetResolutionFeature.setSubFeatures([
    SubFeature(adoptionOfTheBudgetResolutionFeature, CongressionalValue.N.value, label="n"),
    SubFeature(adoptionOfTheBudgetResolutionFeature, CongressionalValue.Y.value, label="y"),
])

physicianFeeFreezeFeature = Feature(3, featureType=DISCRETE, label="physician-fee-freeze",
                                    outLabels={0: 'republican', 1: 'democrat'})
physicianFeeFreezeFeature.setSubFeatures([
    SubFeature(physicianFeeFreezeFeature, CongressionalValue.N.value, label="n"),
    SubFeature(physicianFeeFreezeFeature, CongressionalValue.Y.value, label="y"),
])

elSalvadorAidFeature = Feature(4, featureType=DISCRETE, label="el-salvador-aid",
                               outLabels={0: 'republican', 1: 'democrat'})
elSalvadorAidFeature.setSubFeatures([
    SubFeature(elSalvadorAidFeature, CongressionalValue.N.value, label="n"),
    SubFeature(elSalvadorAidFeature, CongressionalValue.Y.value, label="y"),
])

religiousGroupsInSchoolsFeature = Feature(5, featureType=DISCRETE, label="religious-groups-in-schools",
                                          outLabels={0: 'republican', 1: 'democrat'})
religiousGroupsInSchoolsFeature.setSubFeatures([
    SubFeature(religiousGroupsInSchoolsFeature, CongressionalValue.N.value, label="n"),
    SubFeature(religiousGroupsInSchoolsFeature, CongressionalValue.Y.value, label="y"),
])

antiSatelliteTestBanFeature = Feature(6, featureType=DISCRETE, label="anti-satellite-test-ban",
                                      outLabels={0: 'republican', 1: 'democrat'})
antiSatelliteTestBanFeature.setSubFeatures([
    SubFeature(antiSatelliteTestBanFeature, CongressionalValue.N.value, label="n"),
    SubFeature(antiSatelliteTestBanFeature, CongressionalValue.Y.value, label="y"),
])

aidToNicaraguanContrasFeature = Feature(7, featureType=DISCRETE, label="aid-to-nicaraguan-contras",
                                        outLabels={0: 'republican', 1: 'democrat'})
aidToNicaraguanContrasFeature.setSubFeatures([
    SubFeature(aidToNicaraguanContrasFeature, CongressionalValue.N.value, label="n"),
    SubFeature(aidToNicaraguanContrasFeature, CongressionalValue.Y.value, label="y"),
])

mxMissileFeature = Feature(8, featureType=DISCRETE, label="mx-missile",
                           outLabels={0: 'republican', 1: 'democrat'})
mxMissileFeature.setSubFeatures([
    SubFeature(mxMissileFeature, CongressionalValue.N.value, label="n"),
    SubFeature(mxMissileFeature, CongressionalValue.Y.value, label="y"),
])

immigrationFeature = Feature(9, featureType=DISCRETE, label="immigration",
                             outLabels={0: 'republican', 1: 'democrat'})
immigrationFeature.setSubFeatures([
    SubFeature(immigrationFeature, CongressionalValue.N.value, label="n"),
    SubFeature(immigrationFeature, CongressionalValue.Y.value, label="y"),
])

synfuelsCorporationCutbackFeature = Feature(10, featureType=DISCRETE, label="synfuels-corporation-cutback",
                                            outLabels={0: 'republican', 1: 'democrat'})
synfuelsCorporationCutbackFeature.setSubFeatures([
    SubFeature(synfuelsCorporationCutbackFeature, CongressionalValue.N.value, label="n"),
    SubFeature(synfuelsCorporationCutbackFeature, CongressionalValue.Y.value, label="y"),
])

educationSpendingFeature = Feature(11, featureType=DISCRETE, label="education-spending")
educationSpendingFeature.setSubFeatures([
    SubFeature(educationSpendingFeature, CongressionalValue.N.value, label="n"),
    SubFeature(educationSpendingFeature, CongressionalValue.Y.value, label="y"),
])

superfundRightToSueFeature = Feature(12, featureType=DISCRETE, label="superfund-right-to-sue",
                                     outLabels={0: 'republican', 1: 'democrat'})
superfundRightToSueFeature.setSubFeatures([
    SubFeature(superfundRightToSueFeature, CongressionalValue.N.value, label="n"),
    SubFeature(superfundRightToSueFeature, CongressionalValue.Y.value, label="y"),
])

crimeFeature = Feature(13, featureType=DISCRETE, label="crime",
                       outLabels={0: 'republican', 1: 'democrat'})
crimeFeature.setSubFeatures([
    SubFeature(crimeFeature, CongressionalValue.N.value, label="n"),
    SubFeature(crimeFeature, CongressionalValue.Y.value, label="y"),
])

dutyFreeExportsFeature = Feature(14, featureType=DISCRETE, label="duty-free-exports",
                                 outLabels={0: 'republican', 1: 'democrat'})
dutyFreeExportsFeature.setSubFeatures([
    SubFeature(dutyFreeExportsFeature, CongressionalValue.N.value, label="n"),
    SubFeature(dutyFreeExportsFeature, CongressionalValue.Y.value, label="y"),
])

exportAdministrationActSouthAfricaFeature = Feature(15, featureType=DISCRETE,
                                                    label="export-administration-act-south-africa",
                                                    outLabels={0: 'republican', 1: 'democrat'})
exportAdministrationActSouthAfricaFeature.setSubFeatures([
    SubFeature(exportAdministrationActSouthAfricaFeature, CongressionalValue.N.value, label="n"),
    SubFeature(exportAdministrationActSouthAfricaFeature, CongressionalValue.Y.value, label="y"),
])

congressionalFeatures = [
    handicappedInfantsFeature, waterProjectCostSharingFeature, adoptionOfTheBudgetResolutionFeature,
    physicianFeeFreezeFeature, elSalvadorAidFeature, religiousGroupsInSchoolsFeature, antiSatelliteTestBanFeature,
    aidToNicaraguanContrasFeature, mxMissileFeature, immigrationFeature, synfuelsCorporationCutbackFeature,
    educationSpendingFeature, superfundRightToSueFeature, crimeFeature, dutyFreeExportsFeature,
    exportAdministrationActSouthAfricaFeature,
]


def load_congressional_dataset(train_ratio: float) -> tuple:
    """Cette fonction a pour but de lire le dataset Congressional Voting Records

    dataset reference: http://archive.ics.uci.edu/ml/datasets/Congressional+Voting+Records

    Args:
        train_ratio: le ratio des exemples (ou instances) qui vont servir pour l'entrainement,
        le rest des exemples va etre utilisé pour les test_set.

    Retours:
        Cette fonction doit retourner 4 matrices de type Numpy, train_set, train_labels, test_set, et test_labels
        
        - train_set : une matrice numpy qui contient les exemples qui vont etre utilisés pour l'entrainement, chaque
        ligne dans cette matrice représente un example (ou instance) d'entrainement.
        - train_labels : contient les labels (ou les étiquettes) pour chaque example dans train_set, de telle sorte
          que : train_labels[i] est le label (ou l'etiquette) pour l'example train_set[i]
        - test_set : une matrice numpy qui contient les exemples qui vont etre utilisés pour le test_set, chaque
        ligne dans cette matrice représente un example (ou instance) de test_set.
        - test_labels : contient les labels (ou les étiquettes) pour chaque example dans test_set, de telle sorte
          que : test_labels[i] est le label (ou l'etiquette) pour l'example test_set[i]
    """

    random.seed(1)  # Pour avoir les meme nombres aléatoires à chaque initialisation.

    # Vous pouvez utiliser un dictionnaire pour convertir les attributs en numériques 
    # Notez bien qu'on a traduit le symbole "?" pour une valeur numérique
    # Vous pouvez biensur utiliser d'autres valeurs pour ces attributs
    conversion_labels: dict = {'republican': 0, 'democrat': 1,
                               'n': 0, 'y': 1, '?': 2}
    raw_data: list = list()

    # Le fichier du dataset est dans le dossier datasets en attaché 
    with open("datasets/house-votes-84.data") as file:
        for line in file:
            if line:
                line: str = line.replace("\n", "")
                raw_data.append([conversion_labels[element] for element in line.split(",")])

    random.shuffle(raw_data)

    train_group: list = raw_data[:int(len(raw_data) * train_ratio)]
    test_group: list = raw_data[int(len(raw_data) * train_ratio):]

    train: list = [element[1:] for element in train_group]
    test: list = [element[1:] for element in test_group]

    train_labels: list = [element[0] for element in train_group]
    test_labels: list = [element[0] for element in test_group]

    return np.array(train), np.array(train_labels), np.array(test), np.array(test_labels)


# we sett the features for Monks dataset
a1Feature = Feature(0, featureType=DISCRETE, label="a1")
a1Feature.setSubFeatures([
    SubFeature(a1Feature, 1),
    SubFeature(a1Feature, 2),
    SubFeature(a1Feature, 3),
])

a2Feature = Feature(1, featureType=DISCRETE, label="a2")
a2Feature.setSubFeatures([
    SubFeature(a2Feature, 1),
    SubFeature(a2Feature, 2),
    SubFeature(a2Feature, 3),
])

a3Feature = Feature(2, featureType=DISCRETE, label="a3")
a3Feature.setSubFeatures([
    SubFeature(a3Feature, 1),
    SubFeature(a3Feature, 2),
])

a4Feature = Feature(3, featureType=DISCRETE, label="a4")
a4Feature.setSubFeatures([
    SubFeature(a4Feature, 1),
    SubFeature(a4Feature, 2),
    SubFeature(a4Feature, 3),
])

a5Feature = Feature(4, featureType=DISCRETE, label="a5")
a5Feature.setSubFeatures([
    SubFeature(a5Feature, 1),
    SubFeature(a5Feature, 2),
    SubFeature(a5Feature, 3),
    SubFeature(a5Feature, 4),
])

a6Feature = Feature(5, featureType=DISCRETE, label="a6")
a6Feature.setSubFeatures([
    SubFeature(a6Feature, 1),
    SubFeature(a6Feature, 2),
])

MonksFeatures = [
    a1Feature, a2Feature, a3Feature, a4Feature, a5Feature, a6Feature
]


def load_monks_dataset(numero_dataset):
    """Cette fonction a pour but de lire le dataset Monks

    dataset reference: http://archive.ics.uci.edu/ml/datasets/MONK's+Problems
    
    Notez bien que ce dataset est différent des autres d'un point de vue
    exemples entrainement et exemples de tests.
    Pour ce dataset, nous avons 3 différents sous problèmes, et pour chacun
    nous disposons d'un fichier contenant les exemples d'entrainement et 
    d'un fichier contenant les fichiers de tests. Donc nous avons besoin 
    seulement du numéro du sous problème pour charger le dataset.

    Args:
        numero_dataset: lequel des sous problèmes nous voulons charger (1, 2 ou 3 ?)
        par example, si numero_dataset=2, vous devez lire :
        le fichier monks-2.train_set contenant les exemples pour l'entrainement
        et le fichier monks-2.test_set contenant les exemples pour le test_set
        les fichiers sont tous dans le dossier datasets
    Retours:
        Cette fonction doit retourner 4 matrices de type Numpy, train_set, train_labels, test_set, et test_labels
        
        - train_set : une matrice numpy qui contient les exemples qui vont etre utilisés pour l'entrainement, chaque
        ligne dans cette matrice représente un example (ou instance) d'entrainement.
        - train_labels : contient les labels (ou les étiquettes) pour chaque example dans train_set, de telle sorte
          que : train_labels[i] est le label (ou l'etiquette) pour l'example train_set[i]
        
        - test_set : une matrice numpy qui contient les exemples qui vont etre utilisés pour le test_set, chaque
        ligne dans cette matrice représente un example (ou instance) de test_set.
        - test_labels : contient les labels (ou les étiquettes) pour chaque example dans test_set, de telle sorte
          que : test_labels[i] est le label (ou l'etiquette) pour l'example test_set[i]
    """
    assert numero_dataset in {1, 2, 3}, "param: numero_dataset must be in {1, 2, 3}"

    train_raw_data, train_raw_data_labels = extract_raw_data(f'datasets/monks-{numero_dataset}.train',
                                                             class_index=0, index_to_remove=-1, delimiter=' ')
    test_raw_data, test_raw_data_labels = extract_raw_data(f'datasets/monks-{numero_dataset}.test',
                                                           class_index=0, index_to_remove=-1, delimiter=' ')

    train: np.ndarray = np.array(train_raw_data)
    train_labels: np.ndarray = np.array(train_raw_data_labels)

    test: np.ndarray = np.array(test_raw_data)
    test_labels: np.ndarray = np.array(test_raw_data_labels)

    return train, train_labels, test, test_labels


def extract_raw_data(filename: str, class_index: int = -1, conversion_labels=None, randomize: bool = True,
                     index_to_remove=None, delimiter: str = ','):
    if conversion_labels is None:
        conversion_labels = {str(i): i for i in range(10)}

    if index_to_remove is None:
        index_to_remove = []
    elif isinstance(index_to_remove, int):
        index_to_remove = [index_to_remove]

    if class_index not in index_to_remove:
        index_to_remove.append(class_index)

    raw_data: list = list()
    raw_data_labels: list = list()
    with open(filename, 'r') as file:
        lines: list = file.readlines()
        if randomize:
            random.shuffle(lines)
        for line in lines:
            line: str = line.replace('\n', '').strip()
            if not line:
                continue
            try:
                line_vectorized: list = line.split(delimiter)
                if line_vectorized:
                    cls = line_vectorized[class_index]

                    for idx in reversed(sorted(index_to_remove)):
                        line_vectorized.pop(idx)

                    line_data: list = [float(e) for e in line_vectorized]
                    raw_data.append(line_data)
                    raw_data_labels.append(conversion_labels[cls] if cls in conversion_labels else cls)
            except Exception:
                pass
    return raw_data, raw_data_labels


if __name__ == '__main__':
    import util

    train, train_labels, test, test_labels = load_iris_dataset(0.5)
    print(f"Iris dataset shape: {len(train) + len(test)}")
    congressional_dataset = load_congressional_dataset(0.5)
    print(f"Congressional dataset shape: {len(congressional_dataset[0]) + len(congressional_dataset[2])}")
    for i in range(1, 4):
        monksi_dataset = load_monks_dataset(i)
        print(f"Monks ({i}) dataset shape: {len(monksi_dataset[0]) + len(monksi_dataset[2])}")

    datasets = {
        "Iris": load_iris_dataset(0.5),
        "Congressional": load_congressional_dataset(0.5)
    }
    for i in range(1, 4):
        datasets[f"Monks ({i})"] = load_monks_dataset(i)

    betas = {d: util.beta(v, verbose=True) for d, v in datasets.items()}
    print(betas)
