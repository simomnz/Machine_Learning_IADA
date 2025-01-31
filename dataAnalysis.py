import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def dataAnalysis(df: pd.DataFrame) -> str:
    buffer = []
    buffer.append("Informazioni sul DataFrame:")
    buffer.append(f"Numero di righe: {df.shape[0]}, Numero di colonne: {df.shape[1]}\n")
    missing_values = df.isnull().sum()
    missing_percentage = (missing_values / len(df)) * 100
    buffer.append("Valori Mancanti:")
    buffer.extend([f"{col}: {count} ({pct:.2f}%)" 
                  for col, count, pct in zip(missing_values.index, 
                                           missing_values.values, 
                                           missing_percentage.values)])
    buffer.append("")
    buffer.append("Statistiche Descrittive:")
    buffer.append(str(df.describe()))
    buffer.append("\nStatistiche Descrittive (Include tutti i tipi):")
    buffer.append(str(df.describe(include='all')))
    duplicati = df.duplicated().sum()
    buffer.append(f"\nNumero di duplicati: {duplicati}\n")
    buffer.append("Analisi Outlier per colonna:")
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
        
        buffer.extend([
            f"\n{col}:",
            f"  Numero outlier: {outliers}",
            f"  Valore minimo: {df[col].min():.2f} (limite inferiore: {lower_bound:.2f})",
            f"  Valore massimo: {df[col].max():.2f} (limite superiore: {upper_bound:.2f})"
        ])
    
    buffer.append("\nValori Unici per Colonna Categoriale:")
    for col in df.select_dtypes(include=['object']).columns:
        buffer.append(f"\nColonna: {col}")
        buffer.append(str(df[col].value_counts()))
    
    return "\n".join(buffer)

def show_correlation_matrix(df: pd.DataFrame, figsize=(12, 10), 
                          cmap='coolwarm', fmt='.2f'):
    numeric_df = df.select_dtypes(include=['int64', 'float64'])
    correlation_matrix = numeric_df.corr()
    plt.figure(figsize=figsize)
    sns.heatmap(correlation_matrix, 
                annot=True, 
                cmap=cmap, 
                center=0, 
                fmt=fmt, 
                square=True)
    plt.title("Matrice di Correlazione")
    plt.tight_layout()
    plt.show()

def plot_outliers(df: pd.DataFrame, column: str):
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=df[column])
    plt.title(f'Box Plot of {column}')
    plt.show()
