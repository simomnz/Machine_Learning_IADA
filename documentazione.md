# Relazione Progetto Machine Learning
**Corso di Laurea Triennale in Informatica Applicata e Data Analytics**  
**Studenti:** Simone Sulis e Simone Manunza  
**Matricole:** 00194 / 00184  
**Anno Accademico:** 2024/2025

---

## Introduzione
Il presente progetto si propone di analizzare il dataset `ObesityDataSet_raw_and_data_sinthetic.csv` al fine di classificare il livello di obesità degli individui sulla base di variabili fisiche e abitudini alimentari.  
L'obiettivo principale è confrontare diversi modelli di Machine Learning e individuare la combinazione ottimale di tecniche di preprocessing e parametri (tuning) per massimizzare l'accuratezza del sistema predittivo.

---

## Analisi dei Dati
Prima di procedere con l'addestramento dei modelli, è stata condotta un'analisi esplorativa del dataset:
- **Verifica di dati mancanti e duplicati:** Controllo e, se necessario, rimozione di eventuali dati mancanti e duplicati.
- **Rilevamento di outlier:** Applicazione del metodo dell'intervallo interquartile (IQR) per determinare limiti e identificare valori anomali nelle variabili numeriche.
- **Esplorazione delle distribuzioni:** Analisi delle distribuzioni tramite boxplot e matrice di correlazione per le variabili numeriche, e conteggio delle occorrenze per le variabili categoriali.

---

## Preprocessing dei Dati
Il processo di preprocessing prevede:
- **Normalizzazione:** Le variabili numeriche (`Age`, `Height`, `Weight`, `FCVC`, `NCP`, `CH2O`, `FAF`, `TUE`) vengono normalizzate con il `MinMaxScaler`.
- **Codifica:**
  - La variabile target `NObeyesdad` viene codificata tramite `LabelEncoder`.
  - Le variabili categoriali (es. `Gender`, `family_history_with_overweight`, `FAVC`, `CAEC`, `SMOKE`, `SCC`, `CALC`, `MTRANS`) sono trasformate in dummy variables utilizzando `pd.get_dummies(..., drop_first=True)`.
- **Suddivisione del Dataset:**  
  Il dataset viene diviso in training e test (80%-20%) tramite `train_test_split` con `random_state=0`.

---

## Classificatori e Tuning Manuale
La funzione principale `classifiers` gestisce:
- Il preprocessing iniziale dei dati.
- La selezione del classificatore in base al parametro `tipoClassificatore`.  
  I modelli implementati includono:
  - **SVM:** Utilizza un kernel lineare.
  - **Albero Decisionale:** Implementato con `DecisionTreeClassifier` e tuning del parametro `max_depth`.
  - **ANN:** Implementato con `MLPClassifier` a due layer nascosti.
  - **KNN Custom:** Una classe custom (`knn`) che supporta diverse metriche di distanza (es. `manhattan`, `euclidean`, `chebyshev`).  
    La classe è stata adattata ereditando da `BaseEstimator` e `ClassifierMixin` per garantire la compatibilità con GridSearchCV.
  - **Voting Classifier Custom:** Aggrega le predizioni di un Decision Tree, di un KNN (scikit-learn) e di un GaussianNB, combinando le predizioni tramite la moda.
- **Tuning:**  
  Pur essendo effettuato manualmente (es. scelta di `max_depth=10` per l'albero e `distanceMetric="manhattan"` per il KNN), è implementata la possibilità di utilizzare GridSearchCV per ottimizzare gli iperparametri.
- **Pipeline Opzionale:**  
  È possibile integrare ulteriori step di preprocessing (StandardScaler, PCA, RandomOverSampler) tramite una pipeline (usando `ImbPipeline`) per valutare l'impatto combinato di diverse tecniche.

**Esempio di funzione `classifiers`:**
```python
def classifiers(dati: pd.DataFrame, tipoClassificatore: str, tuning: bool = True, usaPipeline: bool = False):
    # Preprocessing dei dati...
    # Selezione del classificatore in base a tipoClassificatore (svm, tree, ann, knn, voting)
    # Tuning tramite GridSearchCV (opzionale)
    return accuratezza
