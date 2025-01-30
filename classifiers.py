import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier


def classifiers(df: pd.DataFrame):

    # preprocessign data
    
    # Min-Max Scaling (portiamo tutte le variabili in un range tra 0 e 1 così che hanno tutte la stessa valenza)
    variabiliNumeriche = ["Age", "Height", "Weight", "FCVC", "NCP", "CH2O", "FAF", "TUE"]
    scaler = MinMaxScaler()
    df[variabiliNumeriche] = scaler.fit_transform(df[variabiliNumeriche])

    label_encoder = LabelEncoder()
    df['NObeyesdad'] = label_encoder.fit_transform(df['NObeyesdad']) # according to gpt è meglio fare così per avere una singola colonna di output


    # One-Hot Encoding 
    variabiliCategoriche = ['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS']
    df = pd.get_dummies(df, columns=variabiliCategoriche, drop_first=True)
    y = df["NObeyesdad"]

    # prendo tutto tranne le y
    X = df.drop(columns=["NObeyesdad"])

    trainX, testX, trainY, testY = train_test_split(X, y, test_size=0.2, random_state=0)
    
    print("if you are reading this you should buy your friend Simone Sulis a flight to japan, and pay him a taxi to go to mount hakina")
    i = input()
    # TODO capire che sampling usare e se fare robe come cross validation e stratified puzzo XD LOL
    # TODO trovare un modo per scegliere kernel, maxdepth e numero di layer e neuroni
    match i:
        case "svm":
            model = SVC(kernel="linear")
        case "tree":
            model = DecisionTreeClassifier(max_depth=10, random_state=0) # 10 è il massimo
        case "ann":
            model = MLPClassifier(hidden_layer_sizes=(10, 10), max_iter=1000) # questo robo è potentissimo, fa 0,95
        case "custom1":
            pass
        case "custom2":
            pass
        case _:
            pass
    
    model.fit(trainX, trainY)
    res = model.predict(testX)
    accuracy = np.mean(res == testY)
    print(f"accuracy = {accuracy}")

