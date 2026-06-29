import subprocess
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Dataset, Data
from torch_geometric.loader import DataLoader
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, RocCurveDisplay, ConfusionMatrixDisplay, precision_score, recall_score, f1_score, roc_auc_score,
    precision_recall_curve, average_precision_score, accuracy_score
)
import shap
from tqdm.notebook import tqdm
import math
import matplotlib
from typing import Callable, Dict, Any, List 
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
import random

k = 10
n = 9
split_seed = 1905
PROJECT_ROOT = Path.cwd()

def set_seed(seed: int = 1905, deterministic_torch: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def sample_hyperparams(model_type: str, n: int, seed: int = 123, kernel ='linear') -> List[Dict[str, Any]]:
    """
    Random search Hyperparameter sampling abhängig vom Modell.
    
    model_type: "GAT", "SVM", "XGBoost", "RF"
    """
    
    rng = np.random.default_rng(seed)
    configs = []

    for _ in range(n):
        if model_type == "GAT":
            cfg = {
                "model": "GAT",
                "dropout": float(rng.choice([0.1, 0.2, 0.3, 0.4])),
                "gat_dropout": float(rng.choice([0.1, 0.2, 0.3, 0.4])),
                "edge_dropout": float(rng.choice([0.1, 0.2, 0.3, 0.4])),
                "lr": float(rng.choice([5e-4, 4e-4, 3e-4])),
                "l1_lambda": float(rng.choice([0, 1e-7, 1e-6])),
                "weight_decay": float(rng.choice([0.02, 0.05, 0.001, 0.003, 0.005])),
                "batch_size": int(rng.choice([16, 32])),
            }

        elif model_type == "SVM":
            if kernel == "linear":
                cfg = {
                    "model": "SVM",
                    "kernel" : "linear",
                    "C": float(rng.choice([5e-3, 1e-2, 5e-2, 1e-1, 1, 10])),
                }
            elif kernel == 'rbf':
                cfg = {
                    "model": "SVM",
                    "kernel" : "rbf",
                    "C": float(rng.choice([10, 50, 100, 300, 500, 1000])),
                    "gamma": float(rng.choice([1e-4, 5e-4, 1e-3, 5e-3, 1e-2]))
                }

        elif model_type == "XGBoost":
            cfg = {
                "model": "XGBoost",
                "eta": float(rng.choice([0.01, 0.03, 0.05, 0.07])),
                "max_depth": int(rng.choice([3, 4, 5, 6])),
                "subsample": float(rng.choice([0.6, 0.8, 1.0])),
                "min_child_weight": int(rng.choice([1, 2, 4, 6])),
                "colsample_bytree": float(rng.choice([0.1, 0.2, 0.3, 0.5])),
                "lambda": float(rng.choice([0.5, 1, 2, 5])),
            }
        elif model_type == "RF":
            cfg = {
                "model": "RF",
                "n_estimators": rng.choice([300, 500, 800]),  
                "max_depth": rng.choice([3, 4, 5, 6]),
                "min_samples_split": rng.choice([10, 20, 30, 50]),
                "min_samples_leaf": rng.choice([5, 10, 15, 20])
            }
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
        configs.append(cfg)
    return configs

#https://www.leoniemonigatti.com/blog/customize-shap-plots.html
default_pos_color = "#ff0051"  
default_neg_color = "#008bfb"
positive_color = "#2563EB"   
negative_color = "#F97316"
def recolor_shap_waterfall():
    fig = plt.gcf()

    for fc in fig.get_children():
        for fcc in fc.get_children():

            if isinstance(fcc, matplotlib.patches.FancyArrow):
                face = matplotlib.colors.to_hex(fcc.get_facecolor()).lower()
                edge = matplotlib.colors.to_hex(fcc.get_edgecolor()).lower()

                if face == default_pos_color or edge == default_pos_color:
                    fcc.set_facecolor(positive_color)
                    fcc.set_edgecolor(positive_color)

                elif face == default_neg_color or edge == default_neg_color:
                    fcc.set_facecolor(negative_color)
                    fcc.set_edgecolor(negative_color)

            elif isinstance(fcc, plt.Text):
                color = matplotlib.colors.to_hex(fcc.get_color()).lower()

                if color == default_pos_color:
                    fcc.set_color(positive_color)

                elif color == default_neg_color:
                    fcc.set_color(negative_color)

#Adjust to used R Version
def removeBatchEffect(train_idx, test_idx, fold, model, outer="False"):
    result = subprocess.run([r"C:/Program Files/R/R-4.5.0/bin/Rscript.exe",
                        "removeBatchEffect.R",
                        train_idx,
                        test_idx,
                        fold,
                        model,
                        outer,
                        str(PROJECT_ROOT)],
                        capture_output=True,
                        check=True,
                        text=True)
    
    if result.returncode != 0:
        print("Return code:", result.returncode)
        print("STDOUT:\n", result.stdout)
        print("STDERR:\n", result.stderr)
        raise RuntimeError("R script failed")

class GeneExpressionDataset(Dataset):
    def __init__(self, exp_matrix_path, conf_matrix_path, threshold):
        super().__init__()
        metadata = pd.read_csv(PROJECT_ROOT / 'datasets/colData.csv', index_col=1)
        self.expr = pd.read_csv(exp_matrix_path, index_col=0).T
        self.scaler = None
        self.labels = (metadata.loc[self.expr.index, 'sex'] == 'male').astype(int)
        self.age = metadata["age"]
        self.country = metadata["country"]
        self.batch = metadata["batch"]
        self.smoking = metadata["smoking_status"]
        self.conf_matrix = pd.read_csv(conf_matrix_path, index_col=0).values
        self.threshold = threshold
        self.edge_index, self.edge_weight = self._create_edge_index_and_weights()
        
    def _create_edge_index_and_weights(self):
        conf = np.copy(self.conf_matrix)
        np.fill_diagonal(conf, 0)

        #upper triangular
        mask = np.triu(conf >= self.threshold, k=1)
        src, dst = np.nonzero(mask)
        weights = conf[src, dst]
        
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_weight = torch.tensor(weights, dtype=torch.float32)
        
        #undirected
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        edge_weight = torch.cat([edge_weight, edge_weight], dim=0)
        
        return edge_index, edge_weight

    def __len__(self):
        return len(self.expr)

    def set_scaler(self, scaler):
        self.scaler = scaler

    def __getitem__(self, idx):
        #x = torch.tensor(self.expr.iloc[idx].values, dtype=torch.float32).unsqueeze(1)
        x = self.expr.iloc[idx].values
        if self.scaler is not None:
            #x = self.scaler.transform(x).squeeze()
            x = self.scaler.transform(x.reshape(1, -1)).squeeze()
        x = torch.tensor(x, dtype=torch.float32).unsqueeze(1)
        y = torch.tensor(self.labels.iloc[idx], dtype=torch.long)
        return Data(x=x, edge_index=self.edge_index, edge_weight=self.edge_weight, y=y)
