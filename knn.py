import numpy as np
from sklearn.metrics.pairwise import euclidean_distances, manhattan_distances

class knn:
    def __init__(self, distanceMetric="euclidean", k=5):
        self.distanceMetric = distanceMetric.lower()
        self.k = k
        self.trainX = None
        self.trainY = None

    def fit(self, trainX, trainY):
        self.trainX = np.array(trainX)
        self.trainY = np.array(trainY)

    def __computeDistance(self, testX):
        # Calcola la distanza in base al metrica scelta
        match self.distanceMetric:
            case "euclidean":
                return euclidean_distances(self.trainX, testX)
            case "chebyshev":
                return np.max(np.abs(self.trainX[:, None, :] - testX[None, :, :]), axis=2)
            case "manhattan":
                return manhattan_distances(self.trainX, testX)

    def predict(self, testX):
        previsioni = []
        distanze = self.__computeDistance(np.array(testX))
        
        for i in range(testX.shape[0]):
            # Trova gli indici dei k vicini più vicini
            kIndici = np.argsort(distanze[:, i])[:self.k]
            kEtichetteVicini = [self.trainY[j] for j in kIndici]
            
            # Conta le occorrenze di ciascuna etichetta tra i vicini
            conteggioEtichette = {}
            for etichetta in kEtichetteVicini:
                if etichetta in conteggioEtichette:
                    conteggioEtichette[etichetta] += 1
                else:
                    conteggioEtichette[etichetta] = 1
            
            # Trova l'etichetta più comune
            piu_comune = max(conteggioEtichette, key=conteggioEtichette.get)
            previsioni.append(piu_comune)

        return np.array(previsioni)
