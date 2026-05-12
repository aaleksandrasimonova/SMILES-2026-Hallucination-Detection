# SMILES-2026 Hallucination Detection — Solution Report

## Reproducibility

### Environment

The solution was developed and tested in Google Colab using:

* Python 3.10
* PyTorch
* transformers
* scikit-learn

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the solution:

```bash
python solution.py
```

The script automatically:

* loads `Qwen/Qwen2.5-0.5B`
* extracts hidden states
* trains the hallucination probe
* evaluates the model
* generates:

  * `results.json`
  * `predictions.csv`

Modified files:

* `aggregation.py`
* `probe.py`
* `splitting.py`

Infrastructure files left unchanged:

* `solution.py`
* `model.py`
* `evaluate.py`

---

# Final Solution

## aggregation.py

The final aggregation pipeline combines:

* multi-layer response-tail pooling
* geometric consistency features
* cross-layer trajectory features

Selected layers:

* 12
* 14
* 16
* 18
* 20

For each selected layer:

* only the final 40% of response tokens are used
* the response tail is divided into early, middle, and late chunks
* middle and late chunk means are concatenated

Additional handcrafted features:

* cosine similarity between chunks
* context-response similarity
* L2 distances
* norm ratios
* response variance
* cross-layer trajectory drift

---

## probe.py

The final probe is a lightweight hybrid ensemble.

Main branch:

* StandardScaler
* PCA (64 components)
* Logistic Regression

Parameters:

* `C = 0.5`
* `class_weight = "balanced"`

Additional lightweight probes are trained independently for each layer chunk.

Final probabilities:

* 90% global probe
* 10% auxiliary layer-wise probes

The classification threshold is selected on the validation split using F1 score.

---

## splitting.py

Evaluation uses stratified 5-fold cross-validation to provide:

* balanced class distribution
* more stable evaluation
* lower metric variance

---

# Final Results

## Final Cross-Validation Results

| Split      | Accuracy |     F1 |  AUROC |
| ---------- | -------: | -----: | -----: |
| Train      |   78.74% | 86.34% | 90.27% |
| Validation |   75.32% | 84.34% | 76.49% |
| Test       |   73.87% | 83.52% | 73.70% |

Additional information:

| Property          |       Value |
| ----------------- | ----------: |
| Feature dimension |        9008 |
| Total samples     |         689 |
| Evaluation folds  |           5 |
| Extraction time   | ~10 seconds |

Primary competition metric:

* **Test AUROC: 73.70%**

---

# Experiments

## Baseline — Final-token Representation

### Setup

* final transformer layer
* final non-padding token
* logistic regression probe

### Results

| Split      | Accuracy |     F1 |  AUROC |
| ---------- | -------: | -----: | -----: |
| Train      |   70.06% | 82.40% | 99.99% |
| Validation |   70.19% | 82.49% | 66.59% |
| Test       |   70.19% | 82.49% | 75.10% |

### Conclusion

The baseline showed unstable generalization and strong overfitting.

---

## Experiment 1 — Large Multi-layer Aggregation

### Setup

Added:

* multi-layer response-tail aggregation
* large chunk-based embeddings

### Results

| Split      | Accuracy |     F1 |   AUROC |
| ---------- | -------: | -----: | ------: |
| Train      |   78.38% | 86.63% | 100.00% |
| Validation |   72.12% | 83.43% |  65.42% |
| Test       |   69.23% | 81.82% |  54.49% |

Feature dimension: `13460`

### Conclusion

The large feature space caused severe overfitting.

---

## Experiment 2 — Tail Chunk Pooling + PCA

### Setup

Added:

* response-tail chunk pooling
* PCA compression
* logistic regression
* stratified 5-fold cross-validation

### Results

| Split      | Accuracy |     F1 |  AUROC |
| ---------- | -------: | -----: | -----: |
| Train      |   76.74% | 84.62% | 85.73% |
| Validation |   74.59% | 83.60% | 74.76% |
| Test       |   73.00% | 82.53% | 72.83% |

Feature dimension: `8980`

### Conclusion

PCA and linear probing improved generalization stability.

---

## Experiment 3 — Context-response Features

### Setup

Added:

* cosine similarity
* L2 distance
* norm ratio
* response drift

### Results

| Split      | Accuracy |     F1 |  AUROC |
| ---------- | -------: | -----: | -----: |
| Train      |   76.88% | 84.96% | 85.74% |
| Validation |   74.41% | 83.72% | 74.76% |
| Test       |   73.15% | 82.92% | 72.76% |

Feature dimension: `9000`

### Conclusion

Geometric consistency features slightly improved stability.

---

## Experiment 4 — PCA Dimensionality Sweep

### Results

| PCA Components | Test Accuracy | Test F1 | Test AUROC |
| -------------: | ------------: | ------: | ---------: |
|             32 |        70.68% |  82.18% |     72.28% |
|             64 |        73.15% |  82.92% |     72.76% |
|             96 |        70.54% |  81.49% |     70.74% |

### Conclusion

Higher PCA dimensionality increased overfitting.

---

## Experiment 5 — MLP Probe

### Setup

Replaced logistic regression with a small MLP classifier.

### Results

| Split      | Accuracy |      F1 |   AUROC |
| ---------- | -------: | ------: | ------: |
| Train      |  100.00% | 100.00% | 100.00% |
| Validation |   69.01% |  79.23% |  67.44% |
| Test       |   69.81% |  80.08% |  67.57% |

### Conclusion

The nonlinear probe heavily overfit on the small dataset.

---

## Experiment 6 — Better Layer Selection

### Setup

Updated layer selection to middle-late semantic layers.

### Results

| Split      | Accuracy |     F1 |  AUROC |
| ---------- | -------: | -----: | -----: |
| Train      |   76.92% | 84.48% | 85.89% |
| Validation |   76.04% | 84.11% | 76.32% |
| Test       |   72.85% | 82.03% | 73.74% |

Feature dimension: `9000`

### Conclusion

Hallucination signals were more informative in middle-late semantic layers.

---

## Experiment 7 — Cross-layer Trajectory Features

### Setup

Added:

* cross-layer cosine similarity
* cross-layer L2 drift

### Results

| Split      | Accuracy |     F1 |  AUROC |
| ---------- | -------: | -----: | -----: |
| Train      |   76.47% | 84.26% | 85.88% |
| Validation |   75.86% | 84.10% | 76.32% |
| Test       |   72.85% | 82.16% | 73.76% |

Feature dimension: `9008`

### Conclusion

Cross-layer trajectory features produced a small but consistent improvement.

---

## Experiment 8 — Full Layer-wise Ensemble

### Setup

Built:

* global probe
* layer-wise probes
* geometry-only probe
* probability averaging ensemble

### Results

| Split      | Accuracy |     F1 |   AUROC |
| ---------- | -------: | -----: | ------: |
| Train      |   90.82% | 94.07% | 100.00% |
| Validation |   74.95% | 84.28% |  74.52% |
| Test       |   71.26% | 82.19% |  71.90% |

Feature dimension: `9008`

### Conclusion

The larger ensemble increased variance and overfitting.

---

## Experiment 9 — Lightweight Hybrid Ensemble

### Setup

Used:

* global PCA logistic regression probe
* lightweight layer-wise probes
* weak probability blending

### Results

| Split      | Accuracy |     F1 |  AUROC |
| ---------- | -------: | -----: | -----: |
| Train      |   78.74% | 86.34% | 90.27% |
| Validation |   75.32% | 84.34% | 76.49% |
| Test       |   73.87% | 83.52% | 73.70% |

### Conclusion

A small auxiliary layer-wise contribution improved generalization while keeping overfitting controlled.

---

# Conclusion

The experiments show that:

* response-tail pooling is more informative than final-token pooling
* middle-late transformer layers contain stronger hallucination signals
* compact linear probes generalize better than neural classifiers on small datasets
* geometric consistency features provide complementary information
* dimensionality reduction and regularization are critical for stable performance
