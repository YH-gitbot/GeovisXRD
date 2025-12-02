#__version__ = "0.1.0"  #实现安装包
#__version__ = "0.2.1"  # 添加啊modeling模块，训练给定超参数
#__version__ = "0.2.2"  # 修复modeling模块中的bug
#__version__ = "0.3.0"  # 添加ShapExplainer模块，实现SHAP值计算和绘图，支持树模型分析
#__version__ = "0.3.1"  # explaining模块中修复bug，支持更多模型的SHAP值计算
#__version__ = "0.3.2"  # explaining模块中修复bug，支持更多类图的绘制
#__version__ = "0.4.0"   # 增加plotting模块，实现四种图的绘制：柱状图，蜂巢图，二维依赖图，三维交互图，且可以自由选择特征
__version__ = "0.5.0"   # 增加causal模块，实现四种因果发现算法：PC，LiNGAM，NOTEARS，SAM

__arthor__ = "Yuanhan"    

from .hallo import hallo

# print("imported pck")

from .modeling import train_model, get_model, save_model
from .explaining import shapexplainer
from .explaining.io import save_shap_results, load_shap_results
from .plotting import plot_summary_bar, plot_beeswarm, plot_pos_neg_ratio
from .plotting import plot_dependence_2d_lowess, plot_dependence_3d_interaction

__all__ = ["hallo", "train_model", "get_model", "save_model",
           "shapexplainer",
           "save_shap_results", "load_shap_results", 
           "plot_summary_bar", "plot_beeswarm", "plot_pos_neg_ratio", 
           "plot_dependence_2d_lowess", "plot_dependence_3d_interaction"]