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
    
    variabiliNumeriche = ["Age", "Height", "Weight", "FCVC", "NCP", "CH2O", "FAF", "TUE"]
    scaler = MinMaxScaler()
    df[variabiliNumeriche] = scaler.fit_transform(df[variabiliNumeriche])

    label_encoder = LabelEncoder()
    df['NObeyesdad'] = label_encoder.fit_transform(df['NObeyesdad'])

    variabiliCategoriche = ['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS']
    df = pd.get_dummies(df, columns=variabiliCategoriche, drop_first=True)
    
    y = df["NObeyesdad"]
    X = df.drop(columns=["NObeyesdad"])

    trainX, testX, trainY, testY = train_test_split(X, y, test_size=0.2, random_state=0)
    
    # Selezione del classificatore in base al tipo specificato
    match classifier_type.lower():
        case "svm":
            model = SVC(kernel="linear")
        case "tree":
            model = DecisionTreeClassifier(max_depth=10, random_state=0)
        case "ann":
            model = MLPClassifier(hidden_layer_sizes=(10, 10), max_iter=1000)
        case "knn":
            model = knn(k=5, distanceMetric="manhattan")
        case "voting":
            model = VotingClassifierCustom(k=5, distanceMetric="manhattan")
        case _:
            raise ValueError("Nessun classificatore valido selezionato.")
    
    model.fit(trainX, trainY)
    
    res = model.predict(testX)
    
    accuracy = np.mean(res == testY)
    
    return accuracy
