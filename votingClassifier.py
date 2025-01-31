import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import mode


# Classificatore multiplo che utilizza DecisionTreeClassifier, KNN e Naive Bayes

class VotingClassifierCustom:

    def __init__(self, k=5, distanceMetric="euclidean"):
        # Inizializzazione dei classificatori con i parametri specificati
        self.decision_tree = DecisionTreeClassifier(random_state=42)
        self.knn = KNeighborsClassifier(n_neighbors=k, metric=distanceMetric)
        self.gaussian_nb = GaussianNB()

    def fit(self, X_train, y_train):
        # Addestramento dei classificatori sui dati di training
        self.decision_tree.fit(X_train, y_train)
        self.knn.fit(X_train, y_train)
        self.gaussian_nb.fit(X_train, y_train)

    def predict(self, X_test):
        # Predizione dei classificatori sui dati di test
        pred_dt = self.decision_tree.predict(X_test)
        pred_knn = self.knn.predict(X_test)
        pred_gnb = self.gaussian_nb.predict(X_test)
        
        # Combinazione delle predizioni in un array
        combined_predictions = np.vstack((pred_dt, pred_knn, pred_gnb)).T
        
        # Calcolo della moda delle predizioni combinate per ottenere la predizione finale
        final_pred, _ = mode(combined_predictions, axis=1) 
        return final_pred.ravel()   
