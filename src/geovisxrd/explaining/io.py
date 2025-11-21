# src/geovisxrd/explaining/io.py

import joblib
import os
from datetime import datetime
import pandas as pd
import numpy as np

def save_shap_results(explainer, shap_values, X, save_dir="shap_results", name_prefix="shap"):

    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    joblib_filename = f"{name_prefix}_{timestamp}.joblib"
    joblib_path = os.path.join(save_dir, joblib_filename)
    
    data_to_save = {
        "explainer": explainer,    
        "shap_values": shap_values,
        "X": X                    
    }
    
    joblib.dump(data_to_save, joblib_path)
    print(f"--- [GeoVisXRD] 完整数据(二进制)已保存至: {joblib_path}")

# 1. 创建专门存放 CSV 的子目录
    csv_dir = os.path.join(save_dir, "csv_data")
    os.makedirs(csv_dir, exist_ok=True)

    csv_filename = f"{name_prefix}_{timestamp}.csv"
    csv_path = os.path.join(csv_dir, csv_filename)
    
    try:
        # --- 🔥 修复 Bug: 定义 values_arr ---
        # SHAP 值有时是 list (分类问题)，有时是 array (回归问题)
        values_arr = shap_values
        if isinstance(shap_values, list):
            # 如果是 list，通常取 index 1 (正类)
            values_arr = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        # -----------------------------------

        # 确定列名
        cols = X.columns if hasattr(X, "columns") else [f"feature_{i}" for i in range(values_arr.shape[1])]
        
        # 创建 DataFrame
        df_shap = pd.DataFrame(values_arr, columns=cols)
        
        # 保存
        df_shap.to_csv(csv_path, index=False)
        print(f"--- [GeoVisXRD] SHAP值表格(CSV)已保存至: {csv_path}")
        
    except Exception as e:
        # 打印出具体错误，方便调试
        print(f"⚠️ CSV 保存失败: {e}")

    # 返回二进制文件的路径，方便后面代码直接加载画图
    return joblib_path


def load_shap_results(filepath):
    """
    加载二进制 SHAP 数据
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"找不到文件: {filepath}")
    
    print(f"--- [GeoVisXRD] 正在加载 SHAP 数据: {filepath} ...")
    data = joblib.load(filepath)
    
    # 返回整个字典，包含 explainer, shap_values, X 等
    return data