import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from statsmodels.nonparametric.smoothers_lowess import lowess

# 全局字体设置
plt.rcParams.update({'font.size': 12})

# =========================================================
# 1. Summary Bar Plot (总体重要性柱状图)
# =========================================================
def plot_summary_bar(shap_values, feature_names, save_path=None, **kwargs):
    """
    绘制 SHAP 重要性柱状图。
    **kwargs: 传递给 shap.summary_plot 的其他参数 (例如 max_display=10)
    """
    print(f"绘制 Summary Bar ({save_path if save_path else '显示'})...")
    
    # 创建新画布，避免和之前的冲突
    plt.figure()
    
    # 核心绘图
    shap.summary_plot(shap_values, feature_names=feature_names, plot_type="bar", show=False, **kwargs)
    
    fig = plt.gcf() # 获取当前图表对象
    
    # 如果有保存路径，则保存；否则只返回对象供后续修改
    if save_path:
        fig.savefig(save_path, dpi=600, bbox_inches='tight')
        plt.close() # 如果保存了就关闭，释放内存
    
    return fig


# =========================================================
# 2. Beeswarm Plot (蜂群图)
# =========================================================
def plot_beeswarm(shap_values, X, feature_names=None, save_path=None, **kwargs):
    """
    绘制 SHAP 蜂群图。
    """
    print(f"绘制 Beeswarm ({save_path if save_path else '显示'})...")
    
    if feature_names is None:
        feature_names = X.columns if hasattr(X, "columns") else [f"F{i}" for i in range(X.shape[1])]

    # 构造 Explanation 对象
    explanation = shap.Explanation(
        values=shap_values,
        base_values=np.mean(shap_values), 
        data=X.values if hasattr(X, "values") else X,
        feature_names=feature_names
    )

    plt.figure(figsize=(10, 8))
    
    # 传递 kwargs 给 shap.plots.beeswarm (比如 max_display=20)
    shap.plots.beeswarm(explanation, show=False, **kwargs)
    
    fig = plt.gcf()
    
    if save_path:
        fig.savefig(save_path, dpi=600, bbox_inches='tight')
        plt.close()
        
    return fig


# =========================================================
# 3. Dependence Plot 
# =========================================================
def plot_dependence_2d_lowess(shap_values, X, 
                              x_feature,           # 【维度1】X轴 (特征数值)
                              y_feature=None,      # 【维度2】Y轴 (SHAP值) - 不填默认等于 x_feature
                              xlim=None, ylim=None, save_path=None, 
                              lower=0.01, upper=0.99):
    """
    2D 依赖图 + 红线 (Lowess) + 去除离群值
    支持 X轴 和 Y轴(SHAP) 自由组合。
    """
    # 1. 默认 Y轴 = X轴 (标准依赖图)
    if y_feature is None:
        y_feature = x_feature
        
    print(f"绘制 2D Lowess: X={x_feature}, Y=SHAP({y_feature}) ...")
    
    # 2. 获取数据
    if isinstance(X, pd.DataFrame):
        try:
            # 获取 X轴数据 (Raw Value)
            x_data = X[x_feature].values
            # 获取 Y轴数据 (SHAP Value)
            col_idx_y = X.columns.get_loc(y_feature)
            y_shap = shap_values[:, col_idx_y]
        except KeyError as e:
            print(f"❌ 找不到特征: {e}")
            return
    else:
        # Numpy 索引模式
        x_data = X[:, int(x_feature)]
        y_shap = shap_values[:, int(y_feature)]
        # 更新显示用的名字
        x_feature = f"Feature {x_feature}"
        y_feature = f"Feature {y_feature}"

    # 3. 去除离群值 (只针对 X轴 过滤，Y轴跟随)
    q_low = np.quantile(x_data, lower)
    q_high = np.quantile(x_data, upper)
    mask = (x_data >= q_low) & (x_data <= q_high)
    
    x_clean = x_data[mask]
    y_clean = y_shap[mask]

    # 4. 绘图
    fig, ax = plt.subplots(dpi=150, figsize=(6, 4))
    
    # 0基准线
    ax.axhline(0, linestyle='--', color='black')
    
    # 蓝色散点
    ax.scatter(x_clean, y_clean, c="#2196F3", s=12, edgecolors="white", lw=0.3, alpha=0.7)
    
    # 红色平滑线 (Lowess)
    try:
        smoothed = lowess(y_clean, x_clean, frac=0.25, return_sorted=True)
        ax.plot(smoothed[:, 0], smoothed[:, 1], color="red", lw=2, label='Trend')
    except Exception as e:
        print(f"⚠️ Lowess 计算失败: {e}")

    # 坐标轴限制
    if ylim: ax.set_ylim(ylim)
    if xlim: ax.set_xlim(xlim)
    
    # 自动生成标签
    ax.set_xlabel(str(x_feature))
    ax.set_ylabel(f"SHAP value for\n{y_feature}")
    
    # 标题 (如果 X 和 Y 不同，标出来)
    if x_feature != y_feature:
        ax.set_title(f"Cross-Dependence: {x_feature} vs SHAP({y_feature})", fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, format='png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   - 已保存: {save_path}")
    else:
        return fig


def plot_dependence_3d_interaction(shap_values, X, 
                                   x_feature,               # 【维度1】X轴: 特征数值
                                   y_feature=None,          # 【维度2】Y轴: SHAP值 (不填默认=x_feature)
                                   interaction_feature=None,# 【维度3】颜色: 特征数值
                                   xlim=None, ylim=None, save_path=None):
    """
    3D 依赖图 + 颜色交互
    X, Y, Color 三个维度完全独立。
    """
    # 1. 默认 Y轴 = X轴
    if y_feature is None:
        y_feature = x_feature

    print(f"绘制 3D Interaction: X={x_feature}, Y=SHAP({y_feature}), Color={interaction_feature} ...")
    
    # 2. 获取数据
    try:
        if isinstance(X, pd.DataFrame):
            # X轴数据
            x_data = X[x_feature].values
            # Y轴数据 (SHAP)
            col_idx_y = X.columns.get_loc(y_feature)
            y_data = shap_values[:, col_idx_y]
            # 颜色数据
            if interaction_feature:
                c_data = X[interaction_feature].values
            else:
                c_data = None
        else:
            # Numpy 模式
            x_data = X[:, int(x_feature)]
            y_data = shap_values[:, int(y_feature)]
            c_data = X[:, int(interaction_feature)] if interaction_feature else None
            # 更新名字
            x_feature = f"Feature {x_feature}"
            y_feature = f"Feature {y_feature}"
            if interaction_feature: interaction_feature = f"Feature {interaction_feature}"

    except KeyError as e:
        print(f"❌ 特征名错误: {e}")
        return

    # 3. 绘图
    fig, ax = plt.subplots(dpi=150, figsize=(6, 4))
    
    # 绘制散点
    if c_data is not None:
        # 有颜色交互：使用 coolwarm 配色 (SHAP 默认风格)
        
        vmin = np.nanpercentile(c_data, 1)
        vmax = np.nanpercentile(c_data, 99)
        
        scatter = ax.scatter(x_data, y_data, c=c_data, 
                             cmap=shap.plots.colors.red_blue,
                             vmin=vmin, vmax=vmax, 
                             s=12, edgecolors="white", lw=0.3, alpha=1.0)
        # 添加颜色条
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label(interaction_feature, fontsize=10)
    else:
        # 无颜色交互：默认蓝色
        ax.scatter(x_data, y_data, c="#2196F3", 
                   s=12, edgecolors="white", lw=0.3, alpha=0.7)

    # 4. 装饰
    ax.axhline(0, linestyle='--', color='black')
    
    if ylim: ax.set_ylim(ylim)
    if xlim: ax.set_xlim(xlim)
    
    ax.set_xlabel(str(x_feature))
    ax.set_ylabel(f"SHAP value for\n{y_feature}")
    
    # 标题
    title = f"Dependence: {x_feature}"
    if x_feature != y_feature:
        title += f" (SHAP: {y_feature})"
    ax.set_title(title, fontsize=10)

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, format='png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"- 已保存: {save_path}")
    else:
        return fig


# =========================================================
# 4. Positive/Negative Ratio Plot
# =========================================================
def plot_pos_neg_ratio(shap_values, feature_names, save_path=None, color_palette=None):
    """
    绘制正负贡献比例图
    """
    print(f"绘制 Ratio Plot ({save_path if save_path else '显示'})...")
    
    # 数据处理
    shap_df = pd.DataFrame(shap_values, columns=feature_names)
    pos_neg_ratios = {}

    for column in shap_df.columns:
        positive_count = (shap_df[column] > 0).sum()
        negative_count = (shap_df[column] < 0).sum()
        total = positive_count + negative_count
        if total > 0:
            pos_neg_ratios[column] = {
                'Positive': positive_count / total, 
                'Negative': negative_count / total
            }

    ratios_df = pd.DataFrame(pos_neg_ratios).T
    filtered_df = ratios_df[ratios_df['Positive'] > 0].sort_values('Positive', ascending=True)

    if color_palette is None:
        color_palette = ['#9acffa', '#aeb5ba']

    fig, ax = plt.subplots(figsize=(10, max(6, len(filtered_df)*0.3)))
    filtered_df.plot(kind='barh', stacked=True, color=color_palette, ax=ax, width=0.8, edgecolor='none')

    ax.set_title('Positive and Negative Ratios per Feature')
    ax.set_xlabel('Ratio')
    ax.set_xlim(0, 1)
    ax.grid(axis='x', linestyle='--', alpha=0.4)
    ax.legend(['Positive', 'Negative'], loc='lower right', frameon=False)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, format='png', dpi=300)
        plt.close()
        
    return ax

# =========================================================
# 5. causal plotting utils
# =========================================================
import networkx as nx

import networkx as nx
import matplotlib.pyplot as plt
import os

def plot_causal_graph_networkx(nx_graph, threshold=0.0, title="Causal Graph", 
                               save_path=None, show=False):
    """
    绘制因果图。
    新增参数: show (bool) - 是否弹出窗口显示图片。批量运行时建议设为 False。
    """
    print(f"🎨 正在绘制因果图 (Threshold={threshold})...")
    
    # 1. 过滤弱边
    filtered_edges = []
    for u, v, d in nx_graph.edges(data=True):
        weight = d.get('weight', 0)
        if abs(weight) >= threshold:
            filtered_edges.append((u, v, weight))
    
    filtered_graph = nx.DiGraph()
    filtered_graph.add_nodes_from(nx_graph.nodes())
    filtered_graph.add_weighted_edges_from(filtered_edges)

    # 2. 绘图
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(filtered_graph, seed=42)
    
    nx.draw(
        filtered_graph, pos, 
        with_labels=True, 
        node_size=3000, 
        node_color='lightblue',
        font_size=10, 
        font_weight='bold', 
        edge_color='gray',
        arrowsize=20
    )
    
    edge_labels = {(u, v): f"{w:.2f}" for u, v, w in filtered_edges}
    nx.draw_networkx_edge_labels(
        filtered_graph, pos, 
        edge_labels=edge_labels,
        font_color='red'
    )

    plt.title(f"{title} (Threshold={threshold})")
    
    # 3. 保存
    if save_path:
        directory = os.path.dirname(save_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"   - 图片已保存: {save_path}")
        
    # 🔥 关键修改在这里！
    if show:
        plt.show()  # 阻塞，等待用户关闭
    else:
        plt.close() # 不阻塞，直接释放内存，继续跑下一行代码
    
    return filtered_edges

