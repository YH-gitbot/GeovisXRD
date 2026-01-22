GeoVisXRDGeoVisXRD (Geovisual Explainable AI for Causal Discovery) is a specialized Python framework for Recursive Spatial Causal Inference and Non-linear Driver Analysis.It is designed to bridge the gap between predictive performance and causal mechanism understanding in complex geospatial systems, such as soil organic carbon (SOC) distribution or frozen ground dynamics.

🌟 Core FeaturesRecursive ML-Causal Pipeline: Beyond simple $X \rightarrow Y$ modeling, it allows training $X \rightarrow SHAP_{X_i}$ to decipher the factors driving the contribution of key variables.Stability-Optimized Causal Discovery: Implements Jaccard similarity analysis to automatically find the optimal threshold for causal graphs, ensuring structural robustness.Advanced Geospatial Plotting: Custom LOWESS trend curves, 3D interaction plots, and Pos/Neg contribution ratio analysis for scientific publications.One-Click Persistence: Seamlessly save and load models, SHAP values (binary), and causal objects for break-point analysis.

📦 InstallationBash# Recommended installation via GitHub
pip install git+https://github.com/YH-gitbot/GeovisXRD.git
Note for macOS users: Ensure libomp is installed via brew install libomp to support XGBoost.

🚀 Key Workflows1. Recursive Mechanism Deciphering (The Paper's Logic)Identify the most prominent feature and analyze what drives its influence.Pythonfrom geovisxrd.modeling.trainer import train_model
from geovisxrd.explaining.io import save_shap_results, load_shap_results

# Stage 1: Global Prediction (X -> y)
model, _ = train_model("xgb", X, y)
explainer, shap_vals = shapexplainer(model, X)
save_shap_results(explainer, shap_vals, X, name_prefix="layer1")

# Stage 2: Recursive Analysis (X -> SHAP_of_Best_Feature)
# Target the SHAP values of 'MedInc' to see what drives its contribution
new_y = shap_vals[:, X.columns.get_loc("MedInc")]
recursive_model, _ = train_model("xgb", X, new_y)
2. Stability-Aware Causal DiscoveryUse Jaccard Index to find the "Elbow Point" where the causal structure becomes stable.Pythonfrom geovisxrd.causal.discovery import get_causal_model
from geovisxrd.threshold.optimizer import analyze_threshold_stability

# Run Causal Discovery (SAM, PC, LiNGAM, or NOTEARS)
sam_model = get_causal_model("sam", train_epochs=100)
sam_model.fit(data)

# Automatically find the best threshold using Jaccard analysis
stability_df, best_t = analyze_threshold_stability(sam_model)

# Plot the optimized graph
geovisxrd.plot_causal_graph_networkx(sam_model.get_nx_graph(), threshold=best_t)
3. Scientific Visualizations for AnalysisGenerate non-linear response curves and interaction heatmaps.Pythonfrom geovisxrd.plotting.plotting import plot_dependence_2d_lowess, plot_pos_neg_ratio

# Plot LOWESS smoothed response curve to identify thresholds/saturation points
plot_dependence_2d_lowess(shap_vals, X, x_feature="MedInc", y_feature="MedInc")

# Analyze Pos/Neg influence distribution
plot_pos_neg_ratio(shap_vals, X.columns)

📁 Data Persistence (Save & Load)GeoVisXRD ensures your research is reproducible. Every stage can be reloaded without recomputing.Data TypeSave MethodLoad MethodML Modelssave_model(model, "xgb")joblib.load(path)SHAP Datasave_shap_results(e, s, X)load_shap_results(path)Causal Graphmodel.save("graph.pkl")pickle.load(open(path, 'rb'))

📋 RequirementsPython >= 3.9Core: numpy, pandas, xgboost, scikit-learn, shapCausal: cdt, causal-learn, lingam (Optional, based on algorithm choice)Visualization: matplotlib, networkx, statsmodels📄 CitationIf you use this framework in your research, please cite:Chen, C., et al. (2025). Geovisual explainable AI for understanding frozen ground in Qinghai-Tibet Plateau urban region. International Journal of Applied Earth Observation and Geoinformation.