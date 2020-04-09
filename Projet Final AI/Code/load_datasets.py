import random

import numpy as np


def load_iris_dataset(train_ratio: float) -> tuple:
    """Cette fonction a pour but de lire le dataset Iris

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


def load_congressional_dataset(train_ratio: float) -> tuple:
    """Cette fonction a pour but de lire le dataset Congressional Voting Records

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


def load_monks_dataset(numero_dataset):
    """Cette fonction a pour but de lire le dataset Monks
    
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
    print(f"Iris dataset shape: {len(train)+len(test)}")
    congressional_dataset = load_congressional_dataset(0.5)
    print(f"Congressional dataset shape: {len(congressional_dataset[0])+len(congressional_dataset[2])}")
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

