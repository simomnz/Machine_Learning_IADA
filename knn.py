# knn_custom.py
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances, manhattan_distances
from sklearn.base import BaseEstimator, ClassifierMixin

class knn(BaseEstimator, ClassifierMixin):
    def __init__(self, distanceMetric="euclidean", k=5):
        self.distanceMetric = distanceMetric  # Non modificare qui il parametro!
        self.k = k
        self.trainX = None
        self.trainY = None

    def fit(self, trainX, trainY):
        self.trainX = np.array(trainX)
        self.trainY = np.array(trainY)
        return self

    def __computeDistance(self, testX):
        # Convertiamo la metrica in minuscolo solo al momento dell'utilizzo
        metric = self.distanceMetric.lower()
        if metric == "euclidean":
            return euclidean_distances(self.trainX, testX)
        elif metric == "chebyshev":
            return np.max(np.abs(self.trainX[:, None, :] - testX[None, :, :]), axis=2)
        elif metric == "manhattan":
            return manhattan_distances(self.trainX, testX)
        else:
            raise ValueError("Metrica non supportata: {}".format(self.distanceMetric))

    def predict(self, testX):
        previsioni = []
        testX = np.array(testX)
        distanze = self.__computeDistance(testX)
        
        for i in range(testX.shape[0]):
            # Trova gli indici dei k vicini più vicini
            kIndici = np.argsort(distanze[:, i])[:self.k]
            kEtichetteVicini = [self.trainY[j] for j in kIndici]
            
            # Conta le occorrenze di ciascuna etichetta tra i vicini
            conteggioEtichette = {}
            for etichetta in kEtichetteVicini:
                conteggioEtichette[etichetta] = conteggioEtichette.get(etichetta, 0) + 1
            
            # Trova l'etichetta più comune
            comune = max(conteggioEtichette, key=conteggioEtichette.get)
            previsioni.append(comune)

        return np.array(previsioni)
