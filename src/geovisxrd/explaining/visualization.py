# src/geovisxrd/explaining/visualization.py

import shap
import matplotlib.pyplot as plt
import os
import numpy as np

def plot_all_shap_charts(shap_data_dict, save_dir="shap_plots"):
    """
    一键生成所有常见的 SHAP 图表 (蜂群图, 柱状图, 瀑布图)。
    
    参数:
    shap_data_dict: load_shap_results 返回的字典，包含 {"explainer", "shap_values", "X"}
    save_dir: 图片保存的文件夹路径
    """
    # 1. 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 2. 解包数据
    shap_values = shap_data_dict["shap_values"]
    X = shap_data_dict["X"]
    explainer = shap_data_dict["explainer"]  # 取出解释器对象
    
    # --- 数据预处理 ---
    
    # 处理分类模型：如果 shap_values 是 list，取正类 (index 1) 或第 0 个
    if isinstance(shap_values, list):
        shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]

    # 关键步骤：从 explainer 中提取 base_value (基准值)
    base_value = 0
    if hasattr(explainer, "expected_value"):
        val = explainer.expected_value
        # 如果是数组（只有一个数），取出来转成标量
        if isinstance(val, (np.ndarray, list)):
             if np.size(val) == 1:
                 base_value = val.item()
             else:
                 # 多分类情况，取对应类别的基准值(这里简单取第一个)
                 base_value = val[0]
        else:
             base_value = val

    # ---------------------------------------------------------
    # 🔥 核心：构建 shap.Explanation 对象
    # ---------------------------------------------------------
    # 这是为了满足 shap.plots.waterfall 等新版绘图函数的严格要求
    explanation = shap.Explanation(
        values=shap_values,
        base_values=base_value,
        data=X.values if hasattr(X, "values") else X,
        feature_names=X.columns if hasattr(X, "columns") else [f"Feat {i}" for i in range(shap_values.shape[1])]
    )
    # ---------------------------------------------------------

    print(f"--- [GeoVisXRD] 正在生成图表，保存至: {save_dir}/ ---")

    # --- 图表 1: 蜂群图 (Summary Dot) ---
    # 展示特征对预测值的影响方向和强度
    print("1. 正在绘制：蜂群图 (Summary Dot)...")
    plt.figure()
    shap.summary_plot(shap_values, X, show=False)
    plt.savefig(os.path.join(save_dir, "shap_summary_dot.png"), bbox_inches='tight', dpi=300)
    plt.close()

    # --- 图表 2: 柱状图 (Summary Bar) ---
    # 展示特征重要性排名
    print("2. 正在绘制：重要性柱状图 (Summary Bar)...")
    plt.figure()
    shap.summary_plot(shap_values, X, plot_type="bar", show=False)
    plt.savefig(os.path.join(save_dir, "shap_summary_bar.png"), bbox_inches='tight', dpi=300)
    plt.close()

    # --- 图表 3: 瀑布图 (Waterfall) ---
    # 展示单条样本的决策过程。这里默认画第 0 条数据作为示例。
    print("3. 正在绘制：单样本瀑布图 (Waterfall - Sample 0)...")
    try:
        plt.figure()
        # explanation[0] 会自动切片，保留 explanation 对象的属性
        shap.plots.waterfall(explanation[0], show=False)
        plt.savefig(os.path.join(save_dir, "shap_waterfall_sample0.png"), bbox_inches='tight', dpi=300)
        plt.close()
    except Exception as e:
        print(f"⚠️ 瀑布图绘制失败 (可能是版本兼容性问题): {e}")
        plt.close()

    print(f"✅ 所有图表生成完毕！")