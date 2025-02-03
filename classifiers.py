import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from knn import knn
from votingClassifier import VotingClassifierCustom

def classifiers(
    dati: pd.DataFrame,
    tipoClassificatore: str,
    tuning: bool = True,
    usaPipeline: bool = False,
    distanza: str = None,
    kValue: int = None
):
    """
    Addestra un classificatore sul DataFrame 'dati' con preprocessing iniziale.

    Parametri:
      - dati: il DataFrame contenente il dataset.
      - tipoClassificatore: stringa (es. "svm", "tree", "ann", "knn", "voting").
      - tuning: se True, utilizza GridSearchCV per il tuning degli iperparametri.
      - usaPipeline: se True, applica una pipeline di preprocessing (StandardScaler, PCA, RandomOverSampler).
      - distanza: (opzionale) la metrica di distanza da usare per KNN/Voting (default "manhattan" se non specificata).
      - kValue: (opzionale) il numero di vicini da usare per KNN/Voting; se impostato a 0, il tuning sceglie automaticamente tra [3, 5, 7].

    Restituisce:
      - accuratezza: accuratezza del modello sui dati di test.
      
    Nota:
      Se il classificatore è "knn" o "voting" e viene usato il tuning, la funzione stampa a console il valore di k e la metrica scelta.
    """
    
    # Crea una copia dei dati per non modificare l'originale
    datiCopia = dati.copy()
    
    # Preprocessing iniziale: normalizzazione e codifica
    variabiliNumeriche = ["Age", "Height", "Weight", "FCVC", "NCP", "CH2O", "FAF", "TUE"]
    normalizzatore = MinMaxScaler()
    datiCopia[variabiliNumeriche] = normalizzatore.fit_transform(datiCopia[variabiliNumeriche])
    
    codificatoreEtichette = LabelEncoder()
    datiCopia["NObeyesdad"] = codificatoreEtichette.fit_transform(datiCopia["NObeyesdad"])
    
    variabiliCategoriche = ['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS']
    datiCopia = pd.get_dummies(datiCopia, columns=variabiliCategoriche, drop_first=True)
    
    etichette = datiCopia["NObeyesdad"]
    features = datiCopia.drop(columns=["NObeyesdad"])
    
    datiAllenamento, datiTest, etichetteAllenamento, etichetteTest = train_test_split(
        features, etichette, test_size=0.2, random_state=0
    )
    
    # Selezione del classificatore
    tipo = tipoClassificatore.lower()
    if tipo == "svm":
        modello = SVC(kernel="linear")
        grigliaParametri = {"C": [0.1, 1, 10]}
    elif tipo == "tree":
        modello = DecisionTreeClassifier(random_state=0)
        grigliaParametri = {"max_depth": [5, 10, 15, 20]}
    elif tipo == "ann":
        modello = MLPClassifier(max_iter=1000)
        grigliaParametri = {"hidden_layer_sizes": [(10, 10), (20, 20)]}
    elif tipo == "knn":
        # Se kValue è 0 (o None), lascia che il tuning scelga automaticamente
        if kValue is None or kValue == 0:
            modello = knn(distanceMetric=distanza if distanza is not None else "manhattan")
            grigliaParametri = {"k": [3, 5, 7], "distanceMetric": [distanza if distanza is not None else "manhattan"]}
        else:
            modello = knn(distanceMetric=distanza if distanza is not None else "manhattan", k=kValue)
            grigliaParametri = {"k": [kValue], "distanceMetric": [distanza if distanza is not None else "manhattan"]}
    elif tipo == "voting":
        if kValue is None or kValue == 0:
            modello = VotingClassifierCustom(distanceMetric=distanza if distanza is not None else "manhattan")
            grigliaParametri = {"k": [3, 5, 7], "distanceMetric": [distanza if distanza is not None else "manhattan"]}
        else:
            modello = VotingClassifierCustom(distanceMetric=distanza if distanza is not None else "manhattan", k=kValue)
            grigliaParametri = {"k": [kValue], "distanceMetric": [distanza if distanza is not None else "manhattan"]}
    else:
        raise ValueError("Nessun classificatore valido selezionato.")
    
    # Tuning tramite GridSearchCV
    if tuning:
        if usaPipeline:
            passi = [("scaler", StandardScaler()), ("pca", PCA(n_components=0.95)), ("oversampler", RandomOverSampler(random_state=42)), ("clf", modello)]
            pipeline = ImbPipeline(passi)
            grigliaPipeline = {"clf__" + key: grigliaParametri[key] for key in grigliaParametri}
            griglia = GridSearchCV(pipeline, grigliaPipeline, cv=3, scoring="accuracy")
        else:
            griglia = GridSearchCV(modello, grigliaParametri, cv=3, scoring="accuracy")
        griglia.fit(datiAllenamento, etichetteAllenamento)
        migliorModello = griglia.best_estimator_
        # Se il classificatore è knn o voting, stampa il valore di k e la metrica usati
        if tipo in ["knn", "voting"]:
            if usaPipeline:
                clf_finale = migliorModello.named_steps["clf"]
            else:
                clf_finale = migliorModello
            print(f"Valore di k scelto: {clf_finale.k}, Metrica: {clf_finale.distanceMetric}")
    else:
        modello.fit(datiAllenamento, etichetteAllenamento)
        migliorModello = modello
        if tipo in ["knn", "voting"]:
            print(f"Valore di k utilizzato: {modello.k}, Metrica: {modello.distanceMetric}")
    
    risultatiPredizioni = migliorModello.predict(datiTest)
    accuratezza = np.mean(risultatiPredizioni == etichetteTest)
    
    return accuratezza
