# dataAnalysis.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def dataAnalysis(df: pd.DataFrame):
    """
    Esegue l'analisi dei dati e ritorna una stringa con i risultati.
    Eventuali visualizzazioni vengono mostrate direttamente.
    """
    buffer = ""
    
    # Informazioni generali
    buffer += str(df.info()) + "\n"
    buffer += f"Numero di righe: {df.shape[0]}, Numero di colonne: {df.shape[1]}\n\n"

    # Valori mancanti
    missing_values = df.isnull().sum()
    missing_percentage = (missing_values / len(df)) * 100
    buffer += f"Valori Mancanti:\n{missing_values}\n"
    buffer += f"Percentuale (%):\n{missing_percentage}\n\n"

    # Statistiche descrittive
    buffer += "Statistiche Descrittive:\n"
    buffer += str(df.describe()) + "\n\n"
    buffer += "Statistiche Descrittive (Include tutti i tipi):\n"
    buffer += str(df.describe(include='all')) + "\n\n"

    # Duplicati
    duplicati = df.duplicated().sum()
    buffer += f"Numero di duplicati: {duplicati}\n\n"

    # Outlier
    variabiliNumeriche = ["Age", "Height", "Weight", "FCVC", "NCP", "CH2O", "FAF", "TUE"]
    Q1 = df[variabiliNumeriche].quantile(0.25)
    Q3 = df[variabiliNumeriche].quantile(0.75)
    IQR = Q3 - Q1
    # Calcola outliers per colonna numerica
    outlier = {}
    for col in variabiliNumeriche:
        lower = Q1[col] - 1.5 * IQR[col]
        upper = Q3[col] + 1.5 * IQR[col]
        count = ((df[col] < lower) | (df[col] > upper)).sum()
        outlier[col] = count
    buffer += "Outlier per colonna:\n"
    buffer += str(outlier) + "\n\n"

    # Valori unici
    buffer += "Valori Unici per Colonna Categoriale:\n"
    for col in df.select_dtypes(include=['object']).columns:
        buffer += f"Colonna: {col}\n"
        buffer += str(df[col].value_counts()) + "\n\n"

    return buffer

def show_correlation_matrix(df: pd.DataFrame):
    """
    Mostra la matrice di correlazione utilizzando seaborn.
    """
    correlation_matrix = df.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title("Matrice di Correlazione")
    plt.show()
