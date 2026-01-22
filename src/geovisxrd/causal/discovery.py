# src/geovisxrd/causal/discovery.py

import os
import numpy as np
import pandas as pd
import networkx as nx
import pickle
from abc import ABC, abstractmethod

# ==============================================================================
# 1. Helper Functions
# ==============================================================================
def _matrix_to_nx(adj_df):
    """Convert adjacency matrix to NetworkX with weights"""
    G = nx.DiGraph()
    G.add_nodes_from(adj_df.columns)
    for col in adj_df.columns:
        for row in adj_df.index:
            weight = adj_df.loc[row, col]
            if weight != 0:
                G.add_edge(row, col, weight=weight)
    return G

# ==============================================================================
# 2. Abstract Base Class 
# ==============================================================================
class BaseCausalDiscovery(ABC):
    def __init__(self, model_name="unknown", **kwargs):
        self.graph_ = None
        self.nx_graph_ = None
        self.params = kwargs
        self.model_name = model_name  

    @abstractmethod
    def fit(self, data):
        pass

    def get_graph(self):
        return self.graph_

    def get_nx_graph(self):
        """
        Return NetworkX object.
        """
        if self.nx_graph_ is None:
            if self.graph_ is not None:
                self.nx_graph_ = _matrix_to_nx(self.graph_)
            else:
                raise RuntimeError("ERROR: Model not trained")

        self.nx_graph_.graph['model_name'] = self.model_name
        self.nx_graph_.graph['params'] = str(self.params)  
        
        return self.nx_graph_
    
    def save(self, path):
        if self.graph_ is None:
            raise ValueError("ERROR: Model not trained, cannot save.")
        # Auto-create save directory
        directory = os.path.dirname(path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        
        if path.endswith('.csv'):
            self.graph_.to_csv(path)
        elif path.endswith('.pkl'):
            # Save complete NetworkX object with node weights and metadata
            with open(path, 'wb') as f:
                # Ensure saving the latest nx_graph_
                graph_to_save = self.nx_graph_ if self.nx_graph_ else self.get_nx_graph()
                pickle.dump(graph_to_save, f)
        
        print(f"SUCCESS: Causal model saved to: {path}")

# --- Simple Correlation Threshold Model (for Baseline) ---
class ThresholdDiscovery(BaseCausalDiscovery):
    def __init__(self, threshold=0.5, **kwargs):
        super().__init__(model_name="threshold", **kwargs)
        self.threshold = threshold
    
    def fit(self, data):
        print(f"INFO: Threshold Baseline (Pearson Correlation)")
        corr = data.corr().abs()
        adj = (corr > self.threshold).astype(int)
        np.fill_diagonal(adj.values, 0)
        self.graph_ = adj

# ==============================================================================
# 3. Concrete Models 
# ==============================================================================
class SAMDiscovery(BaseCausalDiscovery):
    def __init__(self, **kwargs):
        # Auto-tag name as 'sam'
        super().__init__(model_name="sam", **kwargs)

    def fit(self, data):
        print(f"INFO: SAM (Structural Agnostic Model)")
        try:
            from cdt.causality.graph import SAM
            model = SAM(**self.params)
            
            # cdt returns graph
            self.nx_graph_ = model.predict(data)
            self.graph_ = nx.to_pandas_adjacency(self.nx_graph_)
            print("INFO: SAM training finished.")
        except Exception as e:
            print(f"ERROR: SAM-Algorithm Error: {e}")

class NOTEARSDiscovery(BaseCausalDiscovery):
    def __init__(self, threshold=0.1, **kwargs):
        # Auto-tag name as 'notears'
        super().__init__(model_name="notears", **kwargs)
        self.threshold = threshold

    def fit(self, data):
        print(f"INFO: NOTEARS")
        try:
            from causalnex.structure.notears import from_pandas
            sm = from_pandas(data, w_threshold=self.threshold, **self.params)
            adj = nx.adjacency_matrix(sm).todense()
            self.graph_ = pd.DataFrame(adj, index=data.columns, columns=data.columns)
            self.nx_graph_ = _matrix_to_nx(self.graph_)
            print("INFO: NOTEARS training finished.")
        except Exception as e:
            print(f"ERROR: NOTEARS-Algorithm Error: {e}")
            
class LiNGAMDiscovery(BaseCausalDiscovery):
    def __init__(self, **kwargs):
        # Auto-tag name as 'lingam'
        super().__init__(model_name="lingam", **kwargs)

    def fit(self, data):
        print(f"INFO: LiNGAM")
        try:
            import lingam
            model = lingam.DirectLiNGAM(**self.params)
            model.fit(data)
            self.graph_ = pd.DataFrame(model.adjacency_matrix_, index=data.columns, columns=data.columns)
            self.nx_graph_ = _matrix_to_nx(self.graph_)
            print("INFO: LiNGAM training finished.")
        except Exception as e:
            print(f"ERROR: LiNGAM-Algorithm Error: {e}") 

class PCDiscovery(BaseCausalDiscovery):
    def __init__(self, alpha=0.05, **kwargs):
        # Auto-tag name as 'pc'
        super().__init__(model_name="pc", **kwargs)
        self.alpha = alpha

    def fit(self, data):
        print(f"INFO: PC Algorithm")
        
        try:
            from causallearn.search.ConstraintBased.PC import pc
            from causallearn.utils.cit import fisherz
            cg = pc(data.to_numpy(), self.alpha, indep_test=fisherz, verbose=False)
            self.graph_ = pd.DataFrame(cg.G.graph, index=data.columns, columns=data.columns)
            self.nx_graph_ = _matrix_to_nx(self.graph_)
            print("INFO: PC training finished.")
        except Exception as e:
            print(f"ERROR: PC-Algorithm Error: {e}")

# ==============================================================================
# Factory Function
# ==============================================================================
def get_causal_model(method="sam", **kwargs):
    method = method.lower()
    if method == "threshold": return ThresholdDiscovery(**kwargs)
    if method == "sam": return SAMDiscovery(**kwargs)
    if method == "notears": return NOTEARSDiscovery(**kwargs)
    if method == "lingam": return LiNGAMDiscovery(**kwargs)
    if method == "pc": return PCDiscovery(**kwargs)
    raise ValueError(f"Unknown model: {method}")
