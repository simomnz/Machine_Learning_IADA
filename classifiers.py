import numpy as np
from sklearn.model_selection import train_test_split

def classifiers(df: "pandas.DataFrame"):

    # TODO normalizzare/standardizzare i dati prima di fare questo

    y = df["NObeyesdad"]

    # prendo tutto tranne le y
    X = df.drop(columns=["NObeyesdad"])

    trainX, testX, trainY, testY = train_test_split(X, y, test_size=0.2, random_state=0)
    
    i = input()

    match i:
        case "svm":
            res = svm(trainX, trainY, testX)
        case "tree":
            res = decisionTree(trainX, trainY, testX)
        case "ann":
            res = ann(trainX, trainY, testX)
        case "custom1":
            pass
        case "custom2":
            pass
        case _:
            pass

    print(np.mean(res == testY))


def svm(trainX, trainY, testX):
    pass


def ann(trainX, trainY, testX):
    pass

def decisionTree(trainX, trainY, testX):
    pass
