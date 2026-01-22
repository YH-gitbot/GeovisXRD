# GeoVisXRD

**Geovisual Explainable AI for Causal Discovery**

GeoVisXRD is a specialized Python framework for Recursive Spatial Causal Inference and Non-linear Driver Analysis. It bridges the gap between predictive performance and causal mechanism understanding in complex geospatial systems, such as soil organic carbon (SOC) distribution and frozen ground dynamics.

## Core Features

### 1. Recursive ML–Causal Pipeline
Beyond simple X → Y modeling, GeoVisXRD enables recursive learning such as:
```
X → SHAP(X_i)
```
to decipher the factors driving the contribution of key variables.

### 2. Stability-Optimized Causal Discovery
Implements Jaccard similarity analysis to automatically identify the optimal threshold for causal graphs, ensuring structural robustness and reproducibility.

### 3. Advanced Geospatial Plotting
Provides scientific visualization tools including:
- LOWESS trend curves
- 3D-style interaction plots
- Positive/Negative contribution ratio analysis

### 4. One-Click Persistence
Seamlessly save and load models, SHAP values (binary), and causal objects for break-point and stability analysis.

## Installation

**Recommended installation via GitHub:**
```bash
pip install git+https://github.com/YH-gitbot/GeovisXRD.git
```

**macOS Note:** To enable XGBoost support, install OpenMP:
```bash
brew install libomp
```

## Key Workflows

### 1. Recursive Mechanism Deciphering
```python
from geovisxrd.modeling.trainer import train_model
from geovisxrd.explaining.explainer import shapexplainer
from geovisxrd.explaining.io import save_shap_results

# Stage 1: Global Prediction (X -> y)
model, _ = train_model("xgb", X, y)
explainer, shap_vals = shapexplainer(model, X)
save_shap_results(explainer, shap_vals, X, name_prefix="layer1")

# Stage 2: Recursive Analysis (X -> SHAP of Best Feature)
new_y = shap_vals[:, X.columns.get_loc("MedInc")]
recursive_model, _ = train_model("xgb", X, new_y)
```

### 2. Stability-Aware Causal Discovery
```python
from geovisxrd.causal.discovery import get_causal_model
from geovisxrd.threshold.optimizer import analyze_threshold_stability
from geovisxrd.plotting.plotting import plot_causal_graph_networkx

sam_model = get_causal_model("sam", train_epochs=100)
sam_model.fit(X)

stability_df, best_threshold = analyze_threshold_stability(sam_model)

plot_causal_graph_networkx(
  sam_model.get_nx_graph(),
  threshold=best_threshold
)
```

### 3. Scientific Visualization
```python
from geovisxrd.plotting.plotting import (
  plot_dependence_2d_lowess,
  plot_pos_neg_ratio
)

plot_dependence_2d_lowess(
  shap_vals, X,
  x_feature="MedInc",
  y_feature="MedInc"
)

plot_pos_neg_ratio(shap_vals, X.columns)
```

## Data Persistence

| Type | Save | Load |
|------|------|------|
| **ML Models** | `save_model(model, "xgb")` | `joblib.load(path)` |
| **SHAP Results** | `save_shap_results(e, s, X)` | `load_shap_results(path)` |
| **Causal Graphs** | `model.save("graph.pkl")` | `pickle.load(open(path, "rb"))` |

## Requirements

**Python:** >= 3.9

**Core Dependencies:**
- numpy, pandas, scikit-learn, xgboost, shap

**Causal (Optional):**
- cdt, causal-learn, lingam

**Visualization:**
- matplotlib, networkx, statsmodels

## Citation

Chen, C., et al. (2025). Geovisual explainable AI for understanding frozen ground in Qinghai–Tibet Plateau urban region. *International Journal of Applied Earth Observation and Geoinformation*.

