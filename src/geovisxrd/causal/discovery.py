# src/geovisxrd/causal/discovery.py

import numpy as np
import pandas as pd
import networkx as nx
import pickle
from abc import ABC, abstractmethod

# ==============================================================================
# 1. 辅助函数：统一转为带权重的 NetworkX
# ==============================================================================
def _matrix_to_nx(adj_df):
    """
    将邻接矩阵 (DataFrame) 转换为 NetworkX DiGraph。
    自动将矩阵中的数值作为边的 'weight' 属性。
    """
    G = nx.DiGraph()
    G.add_nodes_from(adj_df.columns)
    
    # 遍历矩阵，添加边和权重
    for col in adj_df.columns:
        for row in adj_df.index:
            weight = adj_df.loc[row, col]
            if weight != 0:
                # 这里的逻辑是: row -> col (根据你具体的矩阵定义调整方向)
                # 通常邻接矩阵定义是 A_ij = 1 代表 i -> j
                G.add_edge(row, col, weight=weight)
    return G

# ==============================================================================
# 2. 抽象基类
# ==============================================================================
class BaseCausalDiscovery(ABC):
    def __init__(self, **kwargs):
        self.graph_ = None      # DataFrame (矩阵)
        self.nx_graph_ = None   # NetworkX (图对象)
        self.params = kwargs

    @abstractmethod
    def fit(self, data):
        pass

    def get_graph(self):
        """返回矩阵"""
        return self.graph_

    def get_nx_graph(self):
        """
        返回 NetworkX 对象。
        🔥 关键：确保不管用什么模型，这里都有值，且边里都有 'weight'
        """
        if self.nx_graph_ is None:
            # 如果子类只生成了矩阵，没生成 nx对象，这里自动转换补救
            if self.graph_ is not None:
                self.nx_graph_ = _matrix_to_nx(self.graph_)
            else:
                raise RuntimeError("模型未训练")
        return self.nx_graph_

# ==============================================================================
# 3. 模型：SAM (本身就返回 NX，特殊处理)
# ==============================================================================
class SAMDiscovery(BaseCausalDiscovery):
    def fit(self, data):
        print(f"   [Model] SAM")
        try:
            from cdt.causality.graph import SAM
            model = SAM(**self.params)
            
            # 1. cdt 直接返回 nx 图 (带权重)
            self.nx_graph_ = model.predict(data)
            
            # 2. 同步生成矩阵
            self.graph_ = nx.to_pandas_adjacency(self.nx_graph_)
            print("   ✅ SAM Finished.")
        except Exception as e:
            print(f"❌ SAM Error: {e}")

# ==============================================================================
# 4. 模型：NOTEARS (返回加权矩阵 -> 转 NX)
# ==============================================================================
class NOTEARSDiscovery(BaseCausalDiscovery):
    def __init__(self, threshold=0.1, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold

    def fit(self, data):
        print(f"   [Model] NOTEARS")
        try:
            from causalnex.structure.notears import from_pandas
            sm = from_pandas(data, w_threshold=self.threshold, **self.params)
            
            # 1. causalnex 得到的是加权矩阵
            adj = nx.adjacency_matrix(sm).todense()
            self.graph_ = pd.DataFrame(adj, index=data.columns, columns=data.columns)
            
            # 2. 🔥 转换成标准的带权重 NX 图
            self.nx_graph_ = _matrix_to_nx(self.graph_)
            
        except ImportError: print("❌ 缺少 causalnex")

# ==============================================================================
# 5. 模型：LiNGAM (返回加权矩阵 -> 转 NX)
# ==============================================================================
class LiNGAMDiscovery(BaseCausalDiscovery):
    def fit(self, data):
        print(f"   [Model] LiNGAM")
        try:
            import lingam
            model = lingam.DirectLiNGAM(**self.params)
            model.fit(data)
            
            # 1. LiNGAM 返回加权矩阵 (系数即权重)
            self.graph_ = pd.DataFrame(model.adjacency_matrix_, index=data.columns, columns=data.columns)
            
            # 2. 🔥 转换成标准的带权重 NX 图
            self.nx_graph_ = _matrix_to_nx(self.graph_)
            
        except ImportError: print("❌ 缺少 lingam")

# ==============================================================================
# 6. 模型：PC (返回 0/1 矩阵 -> 转 NX)
# ==============================================================================
class PCDiscovery(BaseCausalDiscovery):
    def __init__(self, alpha=0.05, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha

    def fit(self, data):
        print(f"   [Model] PC Algorithm")
        try:
            from causallearn.search.ConstraintBased.PC import pc
            from causallearn.utils.Cit import fisherz
            cg = pc(data.to_numpy(), self.alpha, indep_test=fisherz, verbose=False)
            
            # 1. PC 返回 0/1 矩阵 (无权重，只有存在性)
            self.graph_ = pd.DataFrame(cg.G.graph, index=data.columns, columns=data.columns)
            
            # 2. 🔥 转换成标准的 NX 图 (权重默认为 1.0)
            self.nx_graph_ = _matrix_to_nx(self.graph_)
            
        except ImportError: print("❌ 缺少 causal-learn")

# ==============================================================================
# 工厂函数
# ==============================================================================
def get_causal_model(method="sam", **kwargs):
    method = method.lower()
    if method == "sam": return SAMDiscovery(**kwargs)
    if method == "notears": return NOTEARSDiscovery(**kwargs)
    if method == "lingam": return LiNGAMDiscovery(**kwargs)
    if method == "pc": return PCDiscovery(**kwargs)
    raise ValueError(f"Unknown model: {method}")