from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.neural_network import MLPClassifier


class HallucinationProbe(nn.Module):

    def __init__(self) -> None:
        super().__init__()

        # self.pipeline = Pipeline([
        #     ("scaler", StandardScaler()),
        #     ("pca", PCA(n_components=64, random_state=42)),
        #     ("clf", LogisticRegression(
        #         C=0.5,
        #         class_weight="balanced",
        #         max_iter=5000,
        #         random_state=42,
        #     )),
        # ])

        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=64, random_state=42)),
            ("clf", MLPClassifier(
                hidden_layer_sizes=(64,),
                activation="relu",
                alpha=1e-2,
                batch_size=32,
                learning_rate_init=1e-3,
                max_iter=300,
                random_state=42,
            )),
        ])

        self.threshold = 0.5

    def forward(self, x):
        raise NotImplementedError

    def fit(self, X, y):

        self.pipeline.fit(X, y)

        return self

    def fit_hyperparameters(self, X_val, y_val):

        probs = self.predict_proba(X_val)[:, 1]

        thresholds = np.linspace(0.1, 0.9, 81)

        best_t = 0.5
        best_f1 = -1

        for t in thresholds:

            preds = (probs >= t).astype(int)

            f1 = f1_score(y_val, preds)

            if f1 > best_f1:
                best_f1 = f1
                best_t = t

        self.threshold = best_t

        print(f"Best threshold: {best_t:.3f}")
        print(f"Best val F1: {best_f1:.4f}")

        return self

    def predict_proba(self, X):

        probs = self.pipeline.predict_proba(X)[:, 1]

        return np.stack([1 - probs, probs], axis=1)

    def predict(self, X):

        probs = self.predict_proba(X)[:, 1]

        return (probs >= self.threshold).astype(int)