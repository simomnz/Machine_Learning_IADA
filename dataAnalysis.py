import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def dataAnalysis(df: pd.DataFrame):
    buffer = []
    
    for col in df.select_dtypes(include=['int64', 'float64']).columns:
        Q1, Q3 = df[col].quantile([0.25, 0.75])
        IQR = Q3 - Q1
        lower_bound, upper_bound = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
        buffer.extend([
            f"{col}:",
            f"{outliers}",
            f"{df[col].min():.2f} ({lower_bound:.2f})",
            f"{df[col].max():.2f} ({upper_bound:.2f})"
        ])
    
    for col in df.select_dtypes(include=['object']).columns:
        buffer.append(f"{col}")
        buffer.append(str(df[col].value_counts()))
    
    return "\n".join(buffer)

def show_correlation_matrix(df: pd.DataFrame, figsize=(12, 10), cmap='coolwarm', fmt='.2f'):
    plt.figure(figsize=figsize)
    sns.heatmap(df.corr(), annot=True, cmap=cmap, center=0, fmt=fmt, square=True)
    plt.tight_layout()
    plt.show()

def plot_outliers(df: pd.DataFrame, column: str):
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=df[column])
    plt.show()
