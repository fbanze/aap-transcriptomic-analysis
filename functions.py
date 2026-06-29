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
import re

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

#helper functions to gereate tables as in https://www.sciencedirect.com/science/article/pii/S0169260725004808
def extract_mean(cell: str) -> float:
    #split mean
    parts = re.split(r"\s*±\s*", str(cell).strip())
    return float(parts[0])

def latex_escape(s: str) -> str:
    return str(s).replace("_", r"\_")

def get_best_indices_per_fold(df: pd.DataFrame) -> dict:
    best = {}
    for col in df.columns:
        means = df[col].apply(extract_mean)
        best[col] = means.idxmax()
    return best

def generate_table_nachos_tex(df_cv,caption,filename,label,metric,cross_scores: dict):
    
    if df_cv.shape[1] != k:
        raise ValueError(f"Expected k fold columns (F0..Fk). Got {df_cv.shape[1]} columns.")
    if set(df_cv.columns) != {f"F{i}" for i in range(k)}:
        raise ValueError(f"Expected columns named F0..Fk. Got: {list(df_cv.columns)}")
    if set(cross_scores.keys()) != {f"t{i}" for i in range(k)}:
        raise ValueError(f"Expected cross_scores keys t0..tk. Got: {sorted(cross_scores.keys())}")

    best_per_fold = get_best_indices_per_fold(df_cv)
    total_cols = k + 2
    pre_cols = ["c", "c"]
    
    for i in range(k):
        if i % 2 == 0:
            pre_cols.append("c")
        else:
            pre_cols.append(r">{\columncolor{foldshade}}c")
    tab_preamble = " ".join(pre_cols)

    with open(filename, "w", encoding="utf-8") as f:

        f.write("\\begin{table}[ht]\n")
        f.write("\\centering\n")
        f.write("\\small\n")
        f.write("\\setlength{\\tabcolsep}{4pt}\n")
        f.write("\\renewcommand{\\arraystretch}{1.15}\n\n")

        f.write("\\resizebox{\\textwidth}{!}{%\n")
        f.write(f"    \\begin{{tabular}}{{{tab_preamble}}}\n")
        f.write("    \\toprule\n")
        f.write(f"    & \\multicolumn{{{k}}}{{c}}{{Fold reserved for test}} \\\\\n")
        #f.write("    & \\multicolumn{" + str(k) + "}{c}{Fold reserved for test} \\\\\n")
        f.write(f"    \\cmidrule(lr){{3-{total_cols}}}\n")
        f.write("    Hyperparameter & & " + " & ".join([f"$F_{{{i}}}$" for i in range(k)]) + " \\\\\n")
        f.write("    configuration \\\\\n")
        f.write("    \\midrule\n\n")

        # AHPO inner CV
        n_rows = len(df_cv.index)
        f.write(
            f"    \\multirow{{{n_rows}}}{{*}}{{\\rotatebox[origin=c]{{90}}{{\\shortstack{{AHPO/\\\\Cross-validation}}}}}} &\n"
        )

        for r_i, (h_idx, row) in enumerate(df_cv.iterrows()):
            if r_i > 0:
                f.write("    & ")

            f.write(f"${latex_escape(h_idx)}$ & ")

            cells = []
            for col in df_cv.columns:
                val = row[col]
                if best_per_fold[col] == h_idx:
                    val = f"\\best{{{val}}}"
                cells.append(val)

            f.write(" & ".join(cells) + " \\\\\n")

        f.write("    \\midrule\n")

        # Best Line
        f.write("    & Best: $h_y$ & ")
        f.write(" & ".join([f"${latex_escape(best_per_fold[f'F{i}'])}$" for i in range(k)]))
        f.write(" \\\\\n")

        f.write("    \\midrule\n\n")

        # Outer CT loop
        f.write("    \\multirow{2}{*}{\\rotatebox[origin=c]{90}{\\shortstack{Cross-\\\\testing}}} &\n")
        f.write("    " + metric + " & " + " & ".join([f"$t_{{{i}}}$: {cross_scores[f't{i}']:.2f}" for i in range(k)]) + " \\\\\n")

        mean = float(np.mean(list(cross_scores.values())))
        std  = float(np.std(list(cross_scores.values()), ddof=0))  # std over t0..t9
        f.write(f"    \\multicolumn{{{total_cols}}}{{c}}{{Average and standard error {mean:.2f} $\\pm$ {std:.2f}}} \\\\\n")

        f.write("    \\bottomrule\n")
        f.write("    \\end{tabular}\n")
        f.write("}\n\n")
        f.write(f"\\caption{{{caption}}}\n")
        f.write(f"\\label{{{label}}}\n")
        f.write("\\end{table}\n")

    return filename