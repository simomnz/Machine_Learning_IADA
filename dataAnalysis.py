# import time
# import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def dataAnalysis(df: "pandas.DataFrame"): # per non importare pandas gli dò il tipo così (funzionerebbe anche senza, però mi dà fastidio) sborra
    
    
    # Informazioni generali
    print(df.info())
    print(f"Numero di righe: {df.shape[0]}, Numero di colonne: {df.shape[1]}")

    # Valori mancanti
    missing_values = df.isnull().sum()
    missing_percentage = (missing_values / len(df)) * 100
    print(f"Valori Mancanti: {missing_values}, Percentuale (%): {missing_percentage}")

    # Statistiche descrittive
    print(df.describe())
    print(df.describe(include='all'))

    # Distribuzioni
    df.hist(figsize=(10, 8), bins=20)
    plt.tight_layout()
    plt.show()

    # Duplicati
    duplicati = df.duplicated().sum()
    print(f"Numero di duplicati: {duplicati}")
    df = df.drop_duplicates()

    # Outlier (IQR)
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    outlier = ((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).sum()
    print("Outlier per colonna:")
    print(outlier)

    # Matrice di correlazione
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title("Matrice di Correlazione")
    plt.show()

    # Valori unici
    for col in df.select_dtypes(include=['object']).columns:
        print(f"Colonna: {col}")
        print(df[col].value_counts())
        print("\n") # saltiamo due righe

