# evaluation.py

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline

def evaluate_preprocessing_combinations(X, y, classifier):
    
    pipelines = {
        "baseline": Pipeline([("clf", classifier)]),
        "standardizzazione": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", classifier)
        ]),
        "selezione_features": Pipeline([
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=0.95)),
            ("clf", classifier)
        ]),
        "bilanciamento": ImbPipeline([
            ("scaler", StandardScaler()),
            ("oversampler", RandomOverSampler(random_state=42)),
            ("clf", classifier)
        ]),
        "combinata": ImbPipeline([
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=0.95)),
            ("oversampler", RandomOverSampler(random_state=42)),
            ("clf", classifier)
        ])
    }
    
    results = {}
    for name, pipe in pipelines.items():
        scores = cross_val_score(pipe, X, y, cv=5, scoring="accuracy")
        results[name] = scores.mean()
    
    return results
