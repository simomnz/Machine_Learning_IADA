import numpy as np
from sklearn.metrics.pairwise import euclidean_distances, manhattan_distances

class knn:
    def __init__(self, distanceMetric="euclidean", k=5):
        self.distanceMetric = distanceMetric.lower() # setto tutto in lowercase così non abbiamo problemi di maiuscole e minuscole
        self.k = k
        self.trainX = None
        self.trainY = None

    def fit(self, trainX, trainY):
        self.trainX = np.array(trainX)
        self.trainY = np.array(trainY)

    def __computeDistance(self, testX): # è privato
        match self.distanceMetric:
            case "euclidean":
                return euclidean_distances(self.trainX, testX)
            case "chebyshev": # impossibile scrivere chebyshev
                return np.max(np.abs(self.trainX[:, None, :] - testX[None, :, :]), axis=2) # ci ho messo un po' a farlo funzionare
            case "manhattan":
                return manhattan_distances(self.trainX, testX)
        
    def predict(self, testX):
        predictions = []

        distances = self.__computeDistance(np.array(testX))
        predictions = []
        
        for i in range(testX.shape[0]):
            kIndices = np.argsort(distances[:, i])[:self.k]
            kNearestLabels = [self.trainY[j] for j in kIndices]
            # creo un dizionario che uso per contare le label più vicine al nuovo record
            labelCounts = {}
            for label in kNearestLabels:
                if label in labelCounts:
                    labelCounts[label] += 1
                else:
                    labelCounts[label] = 1
            
            most_common = max(labelCounts, key=labelCounts.get)
            predictions.append(most_common)

        return np.array(predictions)
