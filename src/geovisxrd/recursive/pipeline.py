import os
import numpy as np
import pandas as pd
from geovisxrd.modeling.trainer import train_model
from geovisxrd.modeling.save import save_model
from geovisxrd.explaining.explainer import shapexplainer
from geovisxrd.explaining.io import save_shap_results, load_shap_results
from geovisxrd.causal.discovery import get_causal_model
from geovisxrd.threshold.optimizer import analyze_threshold_stability
from geovisxrd.plotting.plotting import plot_causal_graph_networkx

def run_recursive_step(layer_num, previous_shap_path, target_feature_name):
    """
    执行 GeoVisXRD 递归分析。
    修改点：自动在输入文件的同级目录下创建结果文件夹。
    """
    # 1. 自动定位输入文件所在的目录
    # 如果 path 是 "test_outputs/shap/xgb_data/shap_xxx.joblib"
    # input_dir 会变成 "test_outputs/shap/xgb_data"
    input_dir = os.path.dirname(os.path.abspath(previous_shap_path))
    
    # 2. 定义新的子文件夹名称和路径
    new_folder_name = f"layer_{layer_num}_{target_feature_name}"
    step_dir = os.path.join(input_dir, new_folder_name)
    
    # 3. 确保目录存在
    os.makedirs(step_dir, exist_ok=True)

    # --- 以下是业务逻辑 ---
    print(f"--- [GeoVisXRD] 开始第 {layer_num} 层分析: {target_feature_name} ---")
    print(f"结果将保存至: {step_dir}")

    # 加载数据
    saved_data = load_shap_results(previous_shap_path)
    X = saved_data["X"]
    prev_shap_values = saved_data["shap_values"]
    
    # 提取目标 SHAP 值
    feat_idx = list(X.columns).index(target_feature_name)
    new_y = pd.Series(prev_shap_values[:, feat_idx], index=X.index, name=f"SHAP_{target_feature_name}")

    # 建模与保存 (指向新的 step_dir)
    model, metrics = train_model("xgb", X, new_y, n_estimators=100, max_depth=6)
    save_model(model, f"layer{layer_num}_xgb", save_dir=os.path.join(step_dir, "models"))
    
    explainer, shap_values = shapexplainer(model, X, save_path=os.path.join(step_dir, "shap_summary.png"))
    current_shap_path = save_shap_results(explainer, shap_values, X, 
                                          save_dir=os.path.join(step_dir, "shap_data"), 
                                          name_prefix=f"layer{layer_num}")

    # 因果发现 (指向新的 step_dir)
    sam_params = {"nruns": 1, "train_epochs": 100, "nh": 10}
    combined_df = pd.concat([X, new_y.to_frame()], axis=1)
    sam_model = get_causal_model("sam", **sam_params)
    sam_model.fit(combined_df)
    
    _, best_t = analyze_threshold_stability(sam_model, save_path=os.path.join(step_dir, "stability_curve.png"))
    
    plot_causal_graph_networkx(sam_model.get_nx_graph(), threshold=best_t, 
                               title=f"Layer {layer_num} Causal (t={best_t})", 
                               save_path=os.path.join(step_dir, "final_causal_graph.png"))

    print(f"--- [GeoVisXRD] 分析完成。下一层数据索引: {current_shap_path} ---")
    return current_shap_path