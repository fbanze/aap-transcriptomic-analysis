# Installation
## Python Dependencies
```
Python 3.11.14
scanpy
datashader
adjustText
torch
torch_geometric
scikit-learn 1.8.0
ipywidgets
shap
xgboost 3.2.0
GraphSHAP-IQ (GitHub)

```
## R Dependencies
```
R 4.5.0
ggplot2
dplyr
BiocManager 3.21
biomaRt
edgeR
limma
ggrepel
stats
stringr
writexl
readxl
forcats
patchwork
tidyr
scales
shadowtext
```

## Datasets 
```
- GIANT monocyte specific functional network top edges: https://hb.flatironinstitute.org/download

- AAP curated list from PENTACON: https://pentaconhq.org/index.html@q=data.html

- AIDA DataFreezeV2: https://cellxgene.cziscience.com/collections/ced320a1-29f3-47c1-a735-513c7084d508

```

## Scripts
```
1. AIDA_Analysis.ipynb (Single cell Analysis)
2. PseudoBulkAggregation.ipynb (Pseudobulk Aggregation of CD14+ monocytes)
3. CD14+_Pseudobulk_Analysis.Rmd (Analysis on the pseudobulk data of CD14+ monocytes)
4. DGE_Analysis.Rmd (Differential Gene Expression Analysis and GIANT adjacency matrix construction)
5. LinearSVM.ipynb (Support Vector Machine classification with Linear Kernel)
6. RF.ipynb (Random Forest classification)
7. XGBoost.ipynb (EXtreme Gradient Boosting classification)
8. GAT.ipynb (Graph classification with graph attention)
9. functions.py (Python script with helper functions for supervised classification)
```