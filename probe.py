from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class HallucinationProbe:

    def __init__(self):
        self.n_layers = 5
        self.hidden_dim = 896

        self.global_pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=64, random_state=42)),
            ("clf", LogisticRegression(
                C=0.5,
                class_weight="balanced",
                max_iter=5000,
                random_state=42,
            )),
        ])

        self.layer_pipelines = []

        for _ in range(self.n_layers):

            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(
                    C=0.05,
                    class_weight="balanced",
                    max_iter=3000,
                    random_state=42,
                )),
            ])

            self.layer_pipelines.append(pipe)

        self.threshold = 0.5

    def split_features(self, X):
        dense_dim = self.n_layers * 2 * self.hidden_dim

        dense = X[:, :dense_dim]

        layer_chunks = []

        chunk_size = 2 * self.hidden_dim

        for i in range(self.n_layers):

            start = i * chunk_size
            end = start + chunk_size

            layer_chunks.append(dense[:, start:end])

        return dense, layer_chunks

    def fit(self, X, y):
        dense, layer_chunks = self.split_features(X)

        self.global_pipeline.fit(dense, y)
        for pipe, chunk in zip(self.layer_pipelines, layer_chunks):
            pipe.fit(chunk, y)

        return self

    def predict_proba(self, X):
        dense, layer_chunks = self.split_features(X)
        global_probs = self.global_pipeline.predict_proba(dense)[:, 1]
        layer_probs = []
        for pipe, chunk in zip(self.layer_pipelines, layer_chunks):
            p = pipe.predict_proba(chunk)[:, 1]
            layer_probs.append(p)

        layer_mean = np.mean(layer_probs, axis=0)
        final_probs = 0.9 * global_probs + 0.1 * layer_mean

        return np.stack([1 - final_probs, final_probs], axis=1)

    def fit_hyperparameters(self, X_val, y_val):
        probs = self.predict_proba(X_val)[:, 1]
        thresholds = np.linspace(0.1, 0.9, 81)

        best_score = -1
        best_t = 0.5

        for t in thresholds:
            preds = (probs >= t).astype(int)
            score = accuracy_score(y_val, preds)
            if score > best_score:
                best_score = score
                best_t = t

        self.threshold = best_t

        print(f"Best threshold: {best_t:.3f}")
        print(f"Best val accuracy: {best_score:.4f}")

        return self

    def predict(self, X):
        probs = self.predict_proba(X)[:, 1]
        return (probs >= self.threshold).astype(int)