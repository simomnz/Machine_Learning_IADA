# Relazione Progetto Machine Learning
**Corso di Laurea Triennale in Informatica Applicata e Data Analytics**  
**Studenti:** Simone Sulis e Simone Manunza  
**Matricole:** 00194 / 00184  
**Anno Accademico:** 2024/2025

---

## Contenuti

1. [Introduzione](#1-introduzione)
   - [Librerie](#11-librerie)
   - [L’applicazione](#12-lapplicazione)
2. [Analisi dei Dati](#2-analisi-dei-dati)
3. [Classificatori](#3-classificatori)
4. [Tecniche di Pre-Processing](#4-tecniche-di-pre-processing)
   - [Encoding dei Dati](#41-encoding-dei-dati)
   - [Bilanciamento dei Dati](#42-bilanciamento-dei-dati)
5. [Conclusioni](#5-conclusioni)
6. [Sviluppi Futuri](#6-sviluppi-futuri)

---

## 1. Introduzione

In questa relazione analizziamo il dataset **ObesityDataSet_raw_and_data_sinthetic.csv** per la classificazione del livello di obesità di un individuo in base a parametri fisici e abitudini alimentari. L'obiettivo del progetto è confrontare diversi modelli di machine learning per ottenere la miglior accuratezza possibile e determinare la combinazione ottimale di tecniche di pre-processing e classificazione.

### 1.1 Librerie

Abbiamo utilizzato **Python 3.10.4** e le seguenti librerie:

| Librerie          |
|-------------------|
| NumPy             |
| Pandas            |
| Matplotlib        |
| Scikit-learn      |
| Tkinter           |
| Seaborn           |
| Imbalanced-learn  |
| SciPy             |

### 1.2 L’applicazione

L’applicazione è sviluppata tramite una GUI in **Tkinter** che permette di:
- **Analizzare i dati**: visualizzando statistiche descrittive, outlier, matrice di correlazione e istogrammi.
- **Addestrare un modello di classificazione**: scegliendo tra vari modelli (SVM, Decision Tree, ANN, KNN Custom, Voting Classifier Custom).
- **Valutare le performance del pre-processing**: confrontando diverse tecniche di bilanciamento e trasformazione tramite validazione incrociata.

---

## 2. Analisi dei Dati

L’analisi esplorativa del dataset prevede:

- **Verifica dei dati mancanti e duplicati:** Controllo e rimozione di eventuali dati mancanti o duplicati.
- **Rilevamento di outlier:** Utilizzo del metodo dell’Intervallo Interquartile (IQR) per definire limiti e identificare valori anomali nelle variabili numeriche.
- **Esplorazione delle distribuzioni:** Generazione di boxplot, istogrammi e la visualizzazione della matrice di correlazione per comprendere le relazioni interne e la distribuzione delle variabili.

---

## 3. Classificatori

La funzione `classifiers` gestisce l'addestramento dei modelli e il tuning dei parametri. I modelli implementati includono:

- **Support Vector Machine (SVM):**  
  Utilizza un kernel lineare per separare i dati.

- **Albero Decisionale:**  
  Implementato tramite `DecisionTreeClassifier`, con ottimizzazione della profondità per prevenire l'overfitting (es. `max_depth=10`).

- **Artificial Neural Network (ANN):**  
  Implementata con `MLPClassifier` a due layer nascosti, attivati con ReLU e addestrati tramite backpropagation. L'uso di Dropout migliora la generalizzazione.

- **K-Nearest Neighbors (KNN) Custom:**  
  Una classe custom che supporta diverse metriche di distanza (ad es. "manhattan", "euclidean", "chebyshev").  
  **Novità:**  
  - È possibile scegliere la metrica di distanza e impostare il valore di *k* tramite l'interfaccia grafica.  
  - Se il valore di *k* viene lasciato pari a 0, il tuning automatico (GridSearchCV) selezionerà il valore migliore dalla griglia (es. `[3, 5, 7]`) e il valore scelto verrà stampato a console.

- **Voting Classifier Custom:**  
  Combina le predizioni di un Decision Tree, di un KNN (scikit-learn) e di un GaussianNB tramite hard voting.  
  **Novità:**  
  - Anche in questo caso, l'interfaccia consente di scegliere la metrica di distanza e il valore di *k*; se quest'ultimo è impostato a 0, il tuning automatico sceglierà il valore ottimale.

### Tuning e Pipeline

- **Tuning:**  
  È implementata la possibilità di utilizzare GridSearchCV per ottimizzare gli iperparametri dei modelli.
- **Pipeline Opzionale:**  
  È possibile includere una pipeline di pre-processing (StandardScaler, PCA, RandomOverSampler) per migliorare ulteriormente le performance del modello.

---

## 4. Tecniche di Pre-Processing

### 4.1 Encoding dei Dati

Per poter utilizzare i modelli di machine learning, i dati vengono codificati come segue:
- **Label Encoding:**  
  La variabile target `NObeyesdad` viene convertita in valori numerici.
- **One-Hot Encoding:**  
  Le variabili categoriali (es. `Gender`, `family_history_with_overweight`, `FAVC`, `CAEC`, `SMOKE`, `SCC`, `CALC`, `MTRANS`) vengono trasformate in dummy variables (con `drop_first=true` per evitare ridondanze).

### 4.2 Bilanciamento dei Dati

Per migliorare l’addestramento dei modelli, sono state applicate diverse tecniche di bilanciamento:
- **Random Oversampling:**  
  Duplica le istanze della classe minoritaria.
- **Random Undersampling:**  
  Rimuove casualmente alcune istanze della classe maggioritaria.
- **SMOTE:**  
  Genera istanze sintetiche per la classe minoritaria.
- **Tecnica combinata:**  
  Integra sia l’oversampling che l’undersampling per ottenere un bilanciamento ottimale.

| Tecnica                | Accuratezza |
|------------------------|-------------|
| Baseline               | 0.7092      |
| Standardizzazione      | 0.8622      |
| Selezione delle Feature| 0.6386      |
| Bilanciamento          | 0.8627      |
| Combinata              | 0.6471      |

---

## 5. Conclusioni

I risultati ottenuti evidenziano che:
- La **rete neurale artificiale (ANN)** ha raggiunto la migliore performance, con un'accuratezza del **96%**.
- L'**Albero Decisionale** ha ottenuto un'accuratezza del **92%**.
- Il **preprocessing dei dati** (normalizzazione, encoding e bilanciamento) ha avuto un impatto cruciale nel miglioramento delle performance dei modelli.

---

## 6. Sviluppi Futuri

Possibili sviluppi futuri includono:
- **Miglioramento del Voting Classifier:**  
  Sostituire l'attuale metodo di hard voting con soft voting, che considera le probabilità di classe.
- **Estensione dei modelli e metriche:**  
  Integrare ulteriori algoritmi di classificazione e metriche di valutazione (come precision, recall, e F1-score).
- **Ottimizzazione automatica:**  
  Raffinare ulteriormente il tuning degli iperparametri e la scelta automatica di *k* per i classificatori KNN e Voting.

### Aggiornamenti Recenti

- **Scelta Automatica del Valore di k:**  
  Nel modulo dei classificatori, se per KNN o Voting il parametro `kValue` viene impostato a 0, il tuning automatico (GridSearchCV) seleziona il valore ottimale dalla griglia (ad esempio, `[3, 5, 7]`). Il valore di *k* scelto e la metrica di distanza vengono stampati a console.
  
- **Interfaccia Grafica:**  
  L'applicazione Tkinter è stata aggiornata per includere checkbox che permettono di attivare il tuning e l'uso della pipeline, oltre a campi per la scelta della metrica di distanza e il valore di *k* per i classificatori KNN e Voting.

---

Questa relazione offre una panoramica completa del progetto, illustrando come sono state implementate le tecniche di analisi dei dati, il pre-processing, i vari modelli di classificazione (con particolare attenzione al tuning dei parametri) e come l'interfaccia grafica consenta di interagire facilmente con il sistema.
