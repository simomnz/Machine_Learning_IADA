import numpy as np
from sklearn.metrics.pairwise import euclidean_distances, manhattan_distances

# Definisce una classe K-Nearest Neighbors (KNN)
class knn:
    # Inizializza la classe KNN con la metrica di distanza e il numero di vicini (k)
    def __init__(self, distanceMetric="euclidean", k=5):
        self.distanceMetric = distanceMetric.lower()  # Converte la metrica di distanza in minuscolo per evitare problemi di sensibilità al caso
        self.k = k  # Numero di vicini da considerare
        self.trainX = None  # Placeholder per le caratteristiche dei dati di addestramento
        self.trainY = None  # Placeholder per le etichette dei dati di addestramento

    # Adatta il modello KNN con i dati di addestramento
    def fit(self, trainX, trainY):
        self.trainX = np.array(trainX)  # Memorizza le caratteristiche dei dati di addestramento come un array numpy
        self.trainY = np.array(trainY)  # Memorizza le etichette dei dati di addestramento come un array numpy

    # Metodo privato per calcolare le distanze tra i dati di addestramento e i dati di test
    def __computeDistance(self, testX):
        match self.distanceMetric:
            case "euclidean":
                return euclidean_distances(self.trainX, testX)  # Calcola le distanze euclidee
            case "chebyshev":
                return np.max(np.abs(self.trainX[:, None, :] - testX[None, :, :]), axis=2)  # Calcola le distanze di Chebyshev
            case "manhattan":
                return manhattan_distances(self.trainX, testX)  # Calcola le distanze di Manhattan

    # Prevede le etichette per i dati di test
    def predict(self, testX):
        previsioni = []  # Inizializza una lista vuota per memorizzare le previsioni

        distanze = self.__computeDistance(np.array(testX))  # Calcola le distanze tra i dati di addestramento e i dati di test
        
        for i in range(testX.shape[0]):
            kIndici = np.argsort(distanze[:, i])[:self.k]  # Ottiene gli indici dei k vicini più prossimi
            kEtichetteVicini = [self.trainY[j] for j in kIndici]  # Ottiene le etichette dei k vicini più prossimi
            
            # Crea un dizionario per contare le occorrenze di ciascuna etichetta tra i k vicini più prossimi
            conteggioEtichette = {}
            for etichetta in kEtichetteVicini:
                if etichetta in conteggioEtichette:
                    conteggioEtichette[etichetta] += 1
                else:
                    conteggioEtichette[etichetta] = 1
            
            # Determina l'etichetta più comune tra i k vicini più prossimi
            piu_comune = max(conteggioEtichette, key=conteggioEtichette.get)
            previsioni.append(piu_comune)  # Aggiunge l'etichetta più comune alle previsioni

        return np.array(previsioni)  # Restituisce le previsioni come un array numpy
