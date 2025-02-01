## Corso di Laurea Triennale in Informatica Applicata e Data Analytics

# Relazione Progetto Machine Learning
  
**Studenti:** Simone Sulis / Simone Manunza
**Matricole** 00194 / 00184  
**Anno Accademico:** 2023/2024  

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

In questa relazione analizziamo il dataset **ObesityDataSet_raw_and_data_sinthetic.csv** per la classificazione del livello di obesità di un individuo in base a parametri fisici e abitudini alimentari.

L'obiettivo del progetto è confrontare diversi modelli di machine learning per ottenere la miglior accuratezza possibile e determinare la combinazione ottimale di tecniche di pre-processing e classificazione.

### 1.1 Librerie

Abbiamo utilizzato **Python 3.10.4** e le seguenti librerie:

| Librerie          |
|--------------------
| NumPy             |
| Pandas            |
| Matplotlib        |
| Scikit-learn      |
| Tkinter           |
| Seaborn           |
| Imbalanced-learn  |
| SciPy             |


### 1.2 L’applicazione

Abbiamo sviluppato una GUI con **Tkinter** che permette di:
- **Analizzare i dati**: statistiche descrittive, outlier, matrice di correlazione.
- **Addestrare un modello di classificazione** a scelta.
- **Visualizzare le performance** delle tecniche di preprocessing e scegliere la migliore combinazione.

---

## 2. Analisi dei Dati

Il dataset contiene informazioni su diverse variabili. **Principali risultati dell'analisi:**

- **Valori mancanti**: nessun valore mancante.
- **Duplicati**: presenti alcuni duplicati, rimossi in pre-processing.
- **Outlier rilevati**:
  - **Age**: 168
  - **Height**: 1
  - **NCP**: 579
  - **Weight**: 1
- **Correlazione**: peso e altezza hanno una correlazione elevata.

---


## 3. Classificatori

Abbiamo testato diversi modelli:

| Modello | Accuratezza |
|---------|------------|
| **K-Nearest Neighbors (KNN) Custom** | 0.86 |
| **Albero Decisionale** | 0.92 |
| **Artificial Neural Network (ANN)** | 0.96 |
| **Support Vector Machine (SVM)** | 0.86 |
| **Voting Classifier (Custom)** | 0.83 |

### 3.2 Albero Decisionale

Il **Decision Tree** è un modello di classificazione basato su regole gerarchiche, dove ogni nodo rappresenta una decisione basata su una feature del dataset. Abbiamo utilizzato un albero con profondità massima di 10 per evitare l’overfitting, ottenendo un’accuratezza del **92%**.

### 3.3 Artificial Neural Network (ANN)

La **rete neurale artificiale** utilizzata è composta da due layer nascosti con 10 neuroni ciascuno, attivati con **ReLU** e addestrati con **backpropagation**. L’uso di **Dropout** ha migliorato la generalizzazione, permettendo di raggiungere un’accuratezza del **96%**.

### 3.4 Support Vector Machine (SVM)

L’**SVM** è stato implementato con un **kernel lineare**, risultando particolarmente efficace per separare i dati in modo robusto. Ha ottenuto un’accuratezza dell’**86%**, confermando la sua efficacia su dataset con variabili ben distribuite.

### 3.5 Voting Classifier

Abbiamo implementato un **Voting Classifier** che combina i risultati di **Decision Tree, KNN e Naive Bayes**. Questo metodo sfrutta il voto di maggioranza per migliorare la stabilità della predizione. L’**hard voting** è stato utilizzato per assegnare la classe con la maggior frequenza tra i modelli. 

Nonostante il **Voting Classifier** abbia ottenuto un’accuratezza leggermente inferiore (**83%**), il suo utilizzo potrebbe garantire una maggiore robustezza rispetto a singoli modelli, specialmente in contesti con elevata varianza nei dati.

---
## 4. Tecniche di Pre-Processing

### 4.1 Encoding dei Dati

Per poter utilizzare i modelli di machine learning, abbiamo eseguito la codifica delle variabili categoriali:
- **Label Encoding** per la variabile target (**NObeyesdad**), convertendola in valori numerici.
- **One-Hot Encoding** per le variabili categoriali (es. **Gender, family_history_with_overweight, FAVC, CAEC, SMOKE, SCC, CALC, MTRANS**), per evitare che un ordine implicito influisca sul modello.

### 4.2 Bilanciamento dei Dati

Abbiamo applicato diverse tecniche di bilanciamento per migliorare l’addestramento:
- **Random Oversampling**: duplicazione delle istanze della classe minoritaria.
- **Random Undersampling**: rimozione casuale di alcune istanze della classe maggioritaria.
- **SMOTE** (Synthetic Minority Over-sampling Technique): generazione sintetica di nuovi dati appartenenti alla classe minoritaria.
- **Tecnica combinata (Oversampling + Undersampling)** per ottenere il miglior equilibrio possibile tra le classi.

| Tecnica                | Accuratezza |
|------------------------|-------------|
| Baseline              | 0.7092      |
| Standardizzazione     | 0.8622      |
| Selezione Feature     | 0.6386      |
| Bilanciamento        | 0.8627      |
| Combinata            | 0.6471      |

---

## 5. Conclusioni

I risultati ottenuti dimostrano che la **rete neurale artificiale (ANN)** ha raggiunto la miglior performance con un'accuratezza del **96%**, seguita dall'**Albero Decisionale** con **92%**.

Il **preprocessing dei dati** ha giocato un ruolo cruciale nel migliorare le performance. L’uso della **standardizzazione e PCA**, insieme a tecniche di bilanciamento, ha permesso di ottimizzare la classificazione.

---

## 6. Possibili Sviluppi Futuri

- **Miglioramento del Voting Classifier**, sostituendo l'attuale metodo di **Hard Voting** con **Soft Voting**, il quale considera le probabilità di classe per ogni modello invece di basarsi solo sulla maggioranza assoluta delle predizioni.


