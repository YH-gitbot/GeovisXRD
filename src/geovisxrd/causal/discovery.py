# src/geovisxrd/causal/discovery.py

from os import path
import os
import numpy as np
import pandas as pd
import networkx as nx
import pickle
from abc import ABC, abstractmethod

# ==============================================================================
# 1. 辅助函数
# ==============================================================================
def _matrix_to_nx(adj_df):
    """将邻接矩阵转为 NetworkX，带权重"""
    G = nx.DiGraph()
    G.add_nodes_from(adj_df.columns)
    for col in adj_df.columns:
        for row in adj_df.index:
            weight = adj_df.loc[row, col]
            if weight != 0:
                G.add_edge(row, col, weight=weight)
    return G

# ==============================================================================
# 2. 抽象基类 (Base) - 🔥 修改点：增加 model_name
# ==============================================================================
class BaseCausalDiscovery(ABC):
    def __init__(self, model_name="unknown", **kwargs):
        self.graph_ = None
        self.nx_graph_ = None
        self.params = kwargs
        self.model_name = model_name  # <--- 新增：存储模型名字

    @abstractmethod
    def fit(self, data):
        pass

    def get_graph(self):
        return self.graph_

    def get_nx_graph(self):
        """
        返回 NetworkX 对象。
        🔥 关键修改：在返回前，把模型名字写入图的元数据中。
        """
        if self.nx_graph_ is None:
            if self.graph_ is not None:
                self.nx_graph_ = _matrix_to_nx(self.graph_)
            else:
                raise RuntimeError("模型未训练")
        
        # 🔥 注入元数据 (Metadata)
        self.nx_graph_.graph['model_name'] = self.model_name
        self.nx_graph_.graph['params'] = str(self.params) # 顺便把参数也存进去
        
        return self.nx_graph_
    
    def save(self, path):
        if self.graph_ is None:
            raise ValueError("模型尚未训练，无法保存。")
        # 自动创建保存目录
        directory = os.path.dirname(path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        
        if path.endswith('.csv'):
            self.graph_.to_csv(path)
        elif path.endswith('.pkl'):
            # 这里保存的是完整的 NetworkX 对象，包含节点权重和元数据
            with open(path, 'wb') as f:
                # 确保保存的是最新的 nx_graph_
                graph_to_save = self.nx_graph_ if self.nx_graph_ else self.get_nx_graph()
                pickle.dump(graph_to_save, f)
        
        print(f"✅ 因果模型已保存至: {path}")

    
# --- 简单相关系数阈值模型 (用于 Baseline) ---
class ThresholdDiscovery(BaseCausalDiscovery):
    def __init__(self, threshold=0.5, **kwargs):
        super().__init__(model_name="threshold", **kwargs)
        self.threshold = threshold
    def fit(self, data):
        print(f"   [Model] Threshold Baseline (Pearson Correlation)")
        corr = data.corr().abs()
        adj = (corr > self.threshold).astype(int)
        np.fill_diagonal(adj.values, 0)
        self.graph_ = adj

# ==============================================================================
# 3. 具体模型 - 🔥 修改点：初始化时传入 name
# ==============================================================================
class SAMDiscovery(BaseCausalDiscovery):
    def __init__(self, **kwargs):
        # 自动标记名字为 'sam'
        super().__init__(model_name="sam", **kwargs)

    def fit(self, data):
        print(f"   [Model] SAM (Structural Agnostic Model)")
        try:
            from cdt.causality.graph import SAM
            model = SAM(**self.params)
            
            # cdt 返回的图
            self.nx_graph_ = model.predict(data)
            self.graph_ = nx.to_pandas_adjacency(self.nx_graph_)
            print("SAM Training finished.")
        except Exception as e:
            print(f"SAM-Algorithm Error: {e}")

class NOTEARSDiscovery(BaseCausalDiscovery):
    def __init__(self, threshold=0.1, **kwargs):
        # 自动标记名字为 'notears'
        super().__init__(model_name="notears", **kwargs)
        self.threshold = threshold

    def fit(self, data):
        print(f"   [Model] NOTEARS")
        try:
            from causalnex.structure.notears import from_pandas
            sm = from_pandas(data, w_threshold=self.threshold, **self.params)
            adj = nx.adjacency_matrix(sm).todense()
            self.graph_ = pd.DataFrame(adj, index=data.columns, columns=data.columns)
            self.nx_graph_ = _matrix_to_nx(self.graph_)
            print("NOTEARS Training finished.")
        except Exception as e:
            print(f"NOTEARS-Algorithm Error: {e}")

class LiNGAMDiscovery(BaseCausalDiscovery):
    def __init__(self, **kwargs):
        # 自动标记名字为 'lingam'
        super().__init__(model_name="lingam", **kwargs)

    def fit(self, data):
        print(f"   [Model] LiNGAM")
        try:
            import lingam
            model = lingam.DirectLiNGAM(**self.params)
            model.fit(data)
            self.graph_ = pd.DataFrame(model.adjacency_matrix_, index=data.columns, columns=data.columns)
            self.nx_graph_ = _matrix_to_nx(self.graph_)
            print("LiNGAM Training finished.")
        except Exception as e:
            print(f"LiNGAM-Algorithm Error: {e}") 

class PCDiscovery(BaseCausalDiscovery):
    def __init__(self, alpha=0.05, **kwargs):
        # 自动标记名字为 'pc'
        super().__init__(model_name="pc", **kwargs)
        self.alpha = alpha

    def fit(self, data):
        print(f"   [Model] PC Algorithm")
        
        try:
            from causallearn.search.ConstraintBased.PC import pc
            from causallearn.utils.cit import fisherz
            cg = pc(data.to_numpy(), self.alpha, indep_test=fisherz, verbose=False)
            self.graph_ = pd.DataFrame(cg.G.graph, index=data.columns, columns=data.columns)
            self.nx_graph_ = _matrix_to_nx(self.graph_)
            print("PC Training finished.")
        except Exception as e:
            print(f"PC-Algorithm Error: {e}")


# ==============================================================================
# 工厂函数
# ==============================================================================
def get_causal_model(method="sam", **kwargs):
    method = method.lower()
    if method == "threshold": return ThresholdDiscovery(**kwargs)
    if method == "sam": return SAMDiscovery(**kwargs)
    if method == "notears": return NOTEARSDiscovery(**kwargs)
    if method == "lingam": return LiNGAMDiscovery(**kwargs)
    if method == "pc": return PCDiscovery(**kwargs)
    raise ValueError(f"Unknown model: {method}")