import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import os

def calculate_jaccard_similarity(graph_a, graph_b):
    """Calculate the Jaccard similarity between two graphs."""
    edges_a = set(graph_a.edges())
    edges_b = set(graph_b.edges())
    if len(edges_a) == 0 and len(edges_b) == 0: return 1.0
    intersection = len(edges_a.intersection(edges_b))
    union = len(edges_a.union(edges_b))
    return intersection / union if union > 0 else 0

def analyze_threshold_stability(data_input, threshold_range=np.arange(0.1, 1.1, 0.1), save_path=None):
    """
    【GeoVisXRD Deep Optimization Version】
    Supports direct reading of saved model files for analysis.
    data_input: Can be a trained model object or a directly loaded adjacency matrix (DataFrame).
    """
    # 1. Compatibility logic: Extract the original weight matrix from different inputs
    if hasattr(data_input, 'graph_'): # If it's a model object
        raw_adj = data_input.graph_
    elif isinstance(data_input, pd.DataFrame): # If it's a directly loaded CSV matrix
        raw_adj = data_input
    elif isinstance(data_input, nx.DiGraph): # If it's a directly loaded PKL graph object
        raw_adj = nx.to_pandas_adjacency(data_input)
    else:
        raise ValueError("ERROR: Unsupported input format. Please provide a model, DataFrame, or nx.DiGraph.")

    print(f"INFO: Extracting weights from the saved model and performing dynamic threshold segmentation...")
    
    results = []
    prev_graph = None

    # 2. Core loop: Segment in memory and calculate adjacency stability
    for th in threshold_range:
        th_val = round(th, 1)
        # Dynamic filtering
        temp_adj = raw_adj.copy()
        temp_adj[temp_adj.abs() < th_val] = 0
        current_graph = nx.from_pandas_adjacency(temp_adj, create_using=nx.DiGraph)
        
        if prev_graph is not None:
            j_score = calculate_jaccard_similarity(prev_graph, current_graph)
            results.append({"Threshold": th_val, "Jaccard": j_score})
            print(f"   - [t={th_val} vs {round(th_val-0.1, 1)}] Stability: {j_score:.4f}")
        
        prev_graph = current_graph

    df_res = pd.DataFrame(results)

    # 3. Plotting logic (replicating Figure 13a from the paper)
    plt.figure(figsize=(7, 6))
    plt.plot(df_res['Threshold'], df_res['Jaccard'], marker='o', color='#1f77b4', linewidth=2, label='Jaccard index')
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
        print(f"INFO: Curve chart generated: {save_path}")

    return df_res, best_th