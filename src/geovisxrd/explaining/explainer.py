# src/geovisxrd/shap_explain.py

import shap
import matplotlib.pyplot as plt
import xgboost
import pandas as pd
import numpy as np     
from tqdm import tqdm

def _get_smart_explainer(model, X):

    model_class_name = type(model).__name__.lower()
    
    print(f"--- [GeoVisXRD] 检测到模型类型: {type(model).__name__} ---")

    tree_keywords = ['forest', 'xgb', 'lgbm', 'tree', 'boosting', 'gradientboost']
    if any(k in model_class_name for k in tree_keywords):
        try:
            print("--- [GeoVisXRD] 正在使用 TreeExplainer (树模型专用)...")
            return shap.TreeExplainer(model)
        except Exception as e:
            print(f"TreeExplainer 失败 ({e})，尝试降级方案...")

    linear_keywords = ['linear', 'ridge', 'lasso', 'elastic']
    if any(k in model_class_name for k in linear_keywords):
        try:
            print("--- [GeoVisXRD] 正在使用 LinearExplainer (线性模型专用)...")
            return shap.LinearExplainer(model, X)
        except Exception:
            pass

    print("--- [GeoVisXRD] 未识别特定类型，使用通用 Explainer (可能会比较慢)...")
    background_data = X
    if len(X) > 100:
        background_data = shap.sample(X, 100)
        
    return shap.Explainer(model.predict, background_data)


def shapexplainer(model, X, plot_type="summary", save_path=None,batch_size=1000):
    
    # 1. 创建解释器
    explainer = _get_smart_explainer(model, X)
    print(f"--- [GeoVisXRD] 正在计算 SHAP 值 (共 {len(X)} 条数据)... ---")
    
    # 2. 分批计算 SHAP 值 
    shap_values_list = []
    
    for i in tqdm(range(0, len(X), batch_size), desc="SHAP 计算进度", unit="batch"):
        
        batch_X = X.iloc[i : i + batch_size]
        
        #判断是否为树模型。线性模型不需要check_additivity参数
        if isinstance(explainer, shap.TreeExplainer):
            batch_shap = explainer.shap_values(batch_X, check_additivity=False)
        else:
            batch_shap = explainer.shap_values(batch_X)
        
        shap_values_list.append(batch_shap)
    
    
    shap_values = np.concatenate(shap_values_list, axis=0)
    
    print("--- [GeoVisXRD] 计算完成，正在绘图... ---")
    
    plt.figure(figsize=(10, 6)) 
    
    if plot_type == "summary":
        #点云图
        shap.summary_plot(shap_values, X, show=False)
    elif plot_type == "bar":
        shap.summary_plot(shap_values, X, plot_type="bar", show=False)
    
    # 4. 保存或显示
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"--- [GeoVisXRD] 图片已保存至: {save_path} ---")
    else:
        print("--- [GeoVisXRD] no save path, can not save) ---")
        
    return explainer, shap_values
