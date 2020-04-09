import numpy as np


class Classifier:
    """
	Parent class for every classifier. Abstract class.
	"""

    def __init__(self, **kwargs):
        """
		c'est un Initializer.
		Vous pouvez passer d'autre paramètres au besoin,
		c'est à vous d'utiliser vos propres notations
		"""
        pass

    def train(self, train_set: np.ndarray, train_labels: np.ndarray,
              verbose: bool = True, **kwargs):
        """
		Used to train_set the current model.

		:param train_set: une matrice de type Numpy et de taille nxm, avec
		n : le nombre d'example d'entrainement dans le dataset
		m : le mobre d'attribus (le nombre de caractéristiques)

		:param train_labels : est une matrice numpy de taille nx1
		:param verbose: True To print results else False. (bool)
		:param kwargs: Parameters to pass to child classes.
		"""

        raise NotImplementedError()

    def predict(self, example, label) -> (int, bool):
        """
		Prédire la classe d'un example donné en entrée
		:param example: Matrice de taille 1xm
		:param label: The label of the example.

		:return (prediction, prediction == label). :rtype (int, bool)

		"""
        raise NotImplementedError()

    def test(self, test_set, test_labels, verbose: bool = True, displayArgs: dict = None) \
            -> (np.ndarray, float, float, float):
        """
		c'est la méthode qui va tester votre modèle sur les données de test_set
		l'argument test_set est une matrice de type Numpy et de taille nxm, avec
		n : le nombre d'example de test_set dans le dataset
		m : le mobre d'attribus (le nombre de caractéristiques)

		test_labels : est une matrice numpy de taille nx1

		vous pouvez rajouter d'autres arguments, il suffit juste de
		les expliquer en commentaire

		Faites le test_set sur les données de test_set, et afficher :
		- la matrice de confision (confusion matrix)
		- l'accuracy
		- la précision (precision)
		- le rappel (recall)

		Bien entendu ces tests doivent etre faits sur les données de test_set seulement

		"""
        confusionMatrix: np.ndarray = self.getConfusionMatrix(test_set, test_labels)
        accuracy: float = self.getAccuracy(test_set, test_labels)
        precision = self.getPrecision(test_set, test_labels)
        recall = self.getRecall(test_set, test_labels)

        if verbose:
            if displayArgs is None:
                displayArgs = {"dataSize": len(test_set), "title": "Test results", "preMessage": ""}
            self.displayStats(confusionMatrix, accuracy, precision, recall, dataSize=displayArgs["dataSize"],
                              title=displayArgs["title"], preMessage=displayArgs["preMessage"])

        return confusionMatrix, accuracy, precision, recall

    def getAccuracy(self, test_set: np.ndarray, test_labels: np.ndarray) -> float:
        """
        Retourne le ratio entre le nombre d'instanbce bien classifier et le nombre d'instance total.
        :param test_set: L'ensemble de données test (np.ndarray)
        :param test_labels: L'ensemble des étiquettes des données test (np.ndarray)
        :return: L'accuracy :rtype: float
        """
        accuracy: float = 0
        for idx, exemple in enumerate(test_set):
            prediction, check = self.predict(exemple, test_labels[idx])
            accuracy += int(check)
        return 100 * (accuracy / len(test_set))

    def getConfusionMatrix(self, test_set: np.ndarray, test_labels: np.ndarray) -> np.ndarray:
        """
        Retourn la matrice de confusion sur l'ensemble des données test.
        Reference: https://en.wikipedia.org/wiki/Confusion_matrix

        :param test_set: L'ensemble de données test (np.ndarray)
        :param test_labels: L'ensemble des étiquettes des données test (np.ndarray)
        :return: La matrice de confusion :rtype: np.ndarray
        """
        labels: list = sorted(list(set(test_labels)))
        labelsToCountclassification: dict = {lbl: [0 for _ in labels] for lbl in labels}
        for idx, example in enumerate(test_set):
            prediction, check = self.predict(example, test_labels[idx])
            labelsToCountclassification[test_labels[idx]][labels.index(prediction)] += 1

        confusionMatrix: np.ndarray = np.array([labelsToCountclassification[lbl] for lbl in labels]).transpose()
        return confusionMatrix

    def getPrecision(self, test_set: np.ndarray, test_labels: np.ndarray) -> np.ndarray:
        """
        Retoure la précision associée à chaque classe.

                         (nombre de prediction correctement attribué à la classe i)
        precision_i = -------------------------------------------------------------------
                              (nombre d'étiquettes attribué à la classe i)

        Référence: https://fr.wikipedia.org/wiki/Pr%C3%A9cision_et_rappel

        :param test_set: L'ensemble de données test (np.ndarray)
        :param test_labels: L'ensemble des étiquettes des données test (np.ndarray)
        :return: Vecteur de précision :rtype: np.ndarray
        """
        confusionMatrix: np.ndarray = self.getConfusionMatrix(test_set, test_labels)
        precisionVector: np.ndarray = np.array([vector[idx] / np.sum(vector)
                                                for idx, vector in enumerate(confusionMatrix)])
        return precisionVector

    def getRecall(self, test_set: np.ndarray, test_labels: np.ndarray) -> np.ndarray:
        """
        Retoure le rappel associé à chaque classe.

                         (nombre de prediction correctement attribué à la classe i)
        rappel_i = -------------------------------------------------------------------
                              (nombre de prediction attribué à la classe i)

        Référence: https://fr.wikipedia.org/wiki/Pr%C3%A9cision_et_rappel

        :param test_set: L'ensemble de données test (np.ndarray)
        :param test_labels: L'ensemble des étiquettes des données test (np.ndarray)
        :return: Vecteur de précision :rtype: np.ndarray
        """
        confusionMatrix_T: np.ndarray = self.getConfusionMatrix(test_set, test_labels).transpose()
        recallVector: np.ndarray = np.array([vector[idx] / np.sum(vector)
                                             for idx, vector in enumerate(confusionMatrix_T)])
        return recallVector

    def displayStats(self, confusionMatrix: np.ndarray = None, accuracy: float = None, precision: np.ndarray = None,
                     recall: np.ndarray = None, dataSize: int = None, title: str = "", preMessage: str = ""):
        """
        Affiche les différentes statistique relié à la classification d'un ensemble de données.
        Affiche dans l'ordre:
                                -> Un certain titre
                                -> La taille de l'ensemble de données
                                -> Un message quelconque
                                -> La matrice de confusion
                                -> L'accuracy
                                -> La precision
                                -> La precision moyenne
                                -> Le rappel
                                -> Le rappel moyen

        :param confusionMatrix: La matrice de confusion (np.ndarray)
        :param accuracy: L'accuracy (np.float)
        :param precision: La precision pour chaque classe (np.ndarray)
        :param recall: Le rappel pour chaque classe (np.ndarray)
        :param dataSize: La taille de l'ensemble de données (int)
        :param title: Un certain titre (str)
        :param preMessage: Un certain message (str)
        :return: None
        """
        print((f"\n {title}:" if title else ""),
              f"Data set size: {dataSize}",
              (f"{preMessage}" if preMessage else ""),
              f"Confusion Matrix: \n {confusionMatrix}",
              f"Accuracy: {accuracy:.2f} %",
              f"Precision [%]: {np.array([np.round(p_i*100, 2) for p_i in precision])}",
              f"Mean Precision: {precision.mean()*100:.2f} %",
              f"Recall [%]: {np.array([np.round(r_i*100, 2) for r_i in  recall])}",
              f"Mean Recall: {recall.mean()*100:.2f} %",
              sep='\n')
