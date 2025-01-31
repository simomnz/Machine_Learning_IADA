# evaluation.py

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline

def evaluate_preprocessing_combinations(X, y, classifier):
    
    # Definizione delle diverse pipeline di preprocessing
    pipelines = {
        "baseline": Pipeline([("clf", classifier)]),  # Pipeline senza preprocessing
        "standardizzazione": Pipeline([
            ("scaler", StandardScaler()),  # Standardizzazione dei dati
            ("clf", classifier)
        ]),
        "selezione_features": Pipeline([
            ("scaler", StandardScaler()),  # Standardizzazione dei dati
            ("pca", PCA(n_components=0.95)),  # Selezione delle feature con PCA
            ("clf", classifier)
        ]),
        "bilanciamento": ImbPipeline([
            ("scaler", StandardScaler()),  # Standardizzazione dei dati
            ("oversampler", RandomOverSampler(random_state=42)),  # Bilanciamento delle classi
            ("clf", classifier)
        ]),
        "combinata": ImbPipeline([
            ("scaler", StandardScaler()),  # Standardizzazione dei dati
            ("pca", PCA(n_components=0.95)),  # Selezione delle feature con PCA
            ("oversampler", RandomOverSampler(random_state=42)),  # Bilanciamento delle classi
            ("clf", classifier)
        ])
    }
    
    results = {}
    for name, pipe in pipelines.items():
        # Valutazione delle pipeline con cross-validation
        scores = cross_val_score(pipe, X, y, cv=5, scoring="accuracy")
        results[name] = scores.mean()
    
    return results

