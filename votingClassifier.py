# voting_classifier_custom.py
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import mode
from sklearn.base import BaseEstimator, ClassifierMixin


# Classificatore multiplo che utilizza DecisionTreeClassifier, KNN e Naive Bayes

class VotingClassifierCustom(BaseEstimator, ClassifierMixin):
    def __init__(self, k=5, distanceMetric="euclidean"):
        self.k = k
        self.distanceMetric = distanceMetric
        self.decision_tree = DecisionTreeClassifier(random_state=42)
        self.knn = KNeighborsClassifier(n_neighbors=k, metric=distanceMetric)
        self.gaussian_nb = GaussianNB()

    def fit(self, X_train, y_train):
        self.decision_tree.fit(X_train, y_train)
        self.knn.fit(X_train, y_train)
        self.gaussian_nb.fit(X_train, y_train)
        return self

    def predict(self, X_test):
        pred_dt = self.decision_tree.predict(X_test)
        pred_knn = self.knn.predict(X_test)
        pred_gnb = self.gaussian_nb.predict(X_test)
        
        # Combinazione delle predizioni: calcolo della moda lungo le colonne
        combined_predictions = np.vstack((pred_dt, pred_knn, pred_gnb)).T
        final_pred, _ = mode(combined_predictions, axis=1) 
        return final_pred.ravel()

