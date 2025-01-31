import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def dataAnalysis(df: pd.DataFrame):
    
    # Inizializza il buffer di output
    buffer = ""
    
    # Informazioni generali sul DataFrame
    buffer += "Informazioni sul DataFrame:\n"
    buffer += str(df.info()) + "\n"
    buffer += f"Numero di righe: {df.shape[0]}, Numero di colonne: {df.shape[1]}\n\n"
    
    # Analisi dei valori mancanti
    missing_values = df.isnull().sum()
    missing_percentage = (missing_values / len(df)) * 100
    buffer += f"Valori Mancanti:\n{missing_values}\n"
    buffer += f"Percentuale (%):\n{missing_percentage.round(2)}\n\n"
    
    # Statistiche descrittive del DataFrame
    buffer += "Statistiche Descrittive:\n"
    buffer += str(df.describe()) + "\n\n"
    buffer += "Statistiche Descrittive (Include tutti i tipi):\n"
    buffer += str(df.describe(include='all')) + "\n\n"
    
    # Contare i duplicati nel DataFrame
    duplicati = df.duplicated().sum()
    buffer += f"Numero di duplicati: {duplicati}\n\n"
    
    # Analisi degli outlier per le variabili numeriche
    variabiliNumeriche = df.select_dtypes(include=['int64', 'float64']).columns
    outlier = {}
    
    for col in variabiliNumeriche:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers_count = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
        outlier[col] = {
            'count': outliers_count,
            'min': df[col].min(),
            'max': df[col].max(),
            'lower_bound': lower_bound,
            'upper_bound': upper_bound
        }
    
    # Aggiungere i risultati dell'analisi degli outlier al buffer
    buffer += "Analisi Outlier per colonna:\n"
    for col, stats in outlier.items():
        buffer += f"\n{col}:\n"
        buffer += f"  Numero outlier: {stats['count']}\n"
        buffer += f"  Valore minimo: {stats['min']:.2f} (limite inferiore: {stats['lower_bound']:.2f})\n"
        buffer += f"  Valore massimo: {stats['max']:.2f} (limite superiore: {stats['upper_bound']:.2f})\n"
    
    # Analisi dei valori unici per le colonne categoriali
    buffer += "\nValori Unici per Colonna Categoriale:\n"
    for col in df.select_dtypes(include=['object']).columns:
        buffer += f"\nColonna: {col}\n"
        buffer += str(df[col].value_counts()) + "\n"
    
    return buffer

def show_correlation_matrix(df: pd.DataFrame):

    # Seleziona solo le colonne numeriche
    numeric_df = df.select_dtypes(include=['int64', 'float64'])
    
    # Calcola la matrice di correlazione
    correlation_matrix = numeric_df.corr()
    
    # Visualizza la matrice di correlazione utilizzando una heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f', square=True)
    plt.title("Matrice di Correlazione")
    plt.tight_layout()
    plt.show()