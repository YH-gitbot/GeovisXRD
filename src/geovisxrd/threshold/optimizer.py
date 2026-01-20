import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import os

def calculate_jaccard_similarity(graph_a, graph_b):
    """计算两个图的 Jaccard 相似度"""
    edges_a = set(graph_a.edges())
    edges_b = set(graph_b.edges())
    if len(edges_a) == 0 and len(edges_b) == 0: return 1.0
    intersection = len(edges_a.intersection(edges_b))
    union = len(edges_a.union(edges_b))
    return intersection / union if union > 0 else 0

def analyze_threshold_stability(data_input, threshold_range=np.arange(0.1, 1.1, 0.1), save_path=None):
    """
    【GeoVisXRD 深度优化版】
    支持直接读取保存好的模型文件进行分析。
    data_input: 可以是训练好的 model 对象，也可以是直接加载的邻接矩阵(DataFrame)
    """
    # 1. 兼容逻辑：从不同输入中提取原始权重矩阵
    if hasattr(data_input, 'graph_'): # 如果是 model 对象
        raw_adj = data_input.graph_
    elif isinstance(data_input, pd.DataFrame): # 如果是直接加载的 CSV 矩阵
        raw_adj = data_input
    elif isinstance(data_input, nx.DiGraph): # 如果是直接加载的 PKL 图对象
        raw_adj = nx.to_pandas_adjacency(data_input)
    else:
        raise ValueError("❌ 不支持的输入格式。请传入 model, DataFrame 或 nx.DiGraph")

    print(f"📊 正在从已保存的模型中提取权重并进行动态阈值分割...")
    
    results = []
    prev_graph = None

    # 2. 核心循环：内存中分割并计算相邻稳定性
    for th in threshold_range:
        th_val = round(th, 1)
        # 动态过滤
        temp_adj = raw_adj.copy()
        temp_adj[temp_adj.abs() < th_val] = 0
        current_graph = nx.from_pandas_adjacency(temp_adj, create_using=nx.DiGraph)
        
        if prev_graph is not None:
            j_score = calculate_jaccard_similarity(prev_graph, current_graph)
            results.append({"Threshold": th_val, "Jaccard": j_score})
            print(f"   - [t={th_val} vs {round(th_val-0.1, 1)}] 稳定性: {j_score:.4f}")
        
        prev_graph = current_graph

    df_res = pd.DataFrame(results)

    # 3. 绘图逻辑 (复刻论文 Fig 13a)
    plt.figure(figsize=(7, 6))
    plt.plot(df_res['Threshold'], df_res['Jaccard'], marker='o', color='#1f77b4', linewidth=2, label='jaccard index')
    plt.title("Jaccard Index Stability Analysis", fontsize=12)
    plt.xlabel("Threshold (t)")
    plt.ylabel("Jaccard index")
    plt.ylim(0, 1.05)
    plt.grid(True, linestyle='--', alpha=0.5)
    
    if not df_res.empty:
        best_th = df_res.loc[df_res['Jaccard'].idxmax(), 'Threshold']
        plt.axvline(x=best_th, color='red', linestyle='--', label=f'Peak Stability (t={best_th})')
        plt.legend()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"✅ 曲线图已生成: {save_path}")

    return df_res, best_th