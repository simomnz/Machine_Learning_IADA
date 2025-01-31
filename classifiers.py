# classifiers.py

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier

from knn import knn
from votingClassifier import VotingClassifierCustom

def classifiers(df: pd.DataFrame, classifier_type: str):
    # Preprocessing dei dati

    # Min-Max Scaling (portiamo tutte le variabili in un range tra 0 e 1 così che hanno tutte la stessa valenza)
    variabiliNumeriche = ["Age", "Height", "Weight", "FCVC", "NCP", "CH2O", "FAF", "TUE"]
    scaler = MinMaxScaler()
    df[variabiliNumeriche] = scaler.fit_transform(df[variabiliNumeriche])

    # Label Encoding per la variabile target 'NObeyesdad'
    label_encoder = LabelEncoder()
    df['NObeyesdad'] = label_encoder.fit_transform(df['NObeyesdad']) # secondo GPT è meglio fare così per avere una singola colonna di output

    # One-Hot Encoding per le variabili categoriche
    variabiliCategoriche = ['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS']
    df = pd.get_dummies(df, columns=variabiliCategoriche, drop_first=True)
    
    # Separazione delle feature (X) e della variabile target (y)
    y = df["NObeyesdad"]
    X = df.drop(columns=["NObeyesdad"])

    # Suddivisione del dataset in training set e test set
    trainX, testX, trainY, testY = train_test_split(X, y, test_size=0.2, random_state=0)
    
    # Selezione del classificatore in base al parametro 'classifier_type'
    match classifier_type.lower():
        case "svm":
            model = SVC(kernel="linear")
        case "tree":
            model = DecisionTreeClassifier(max_depth=10, random_state=0) # 10 è il massimo
        case "ann":
            model = MLPClassifier(hidden_layer_sizes=(10, 10), max_iter=1000) # questo modello è potente, può raggiungere 0,95 di accuratezza
        case "knn":
            model = knn(k=5, distanceMetric="manhattan")
        case "voting":
            model = VotingClassifierCustom(k=5, distanceMetric="manhattan")
        case _:
            raise ValueError("Nessun classificatore valido selezionato.")
    
    # Addestramento del modello
    model.fit(trainX, trainY)
    
    # Predizione sui dati di test
    res = model.predict(testX)
    
    # Calcolo dell'accuratezza
    accuracy = np.mean(res == testY)
    
    return accuracy
