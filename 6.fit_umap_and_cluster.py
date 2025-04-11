import numpy as np
import pandas as pd
import umap

import os, sys

import anndata as ad

import datetime
import time

import cloudpickle
import random


######## data paths  
FULL_DEVEL_EMBED_PATH = "private/devel_imputed_embeddings_full.csv"
FULL_DEVEL_INFO_PATH = "private/devel_imputed_full.csv"
FITTED_UMAP_PATH = "private/fitted_UMAP_model.pkl"
FULL_DEVEL_UMAP_CLUSTER_PATH = "private/devel_full_umap_cluster.csv"


## Step 1. load embed/clinical data (private)
embed_devel_data = pd.read_csv(FULL_DEVEL_EMBED_PATH, sep=",")
embed_devel_data = embed_devel_data[["f.eid", *["MEAN"+str(i) for i in range(1, 129)]]]
embed_devel_data.index = embed_devel_data["f.eid"]

info_devel_data = pd.read_csv(FULL_DEVEL_INFO_PATH, sep=",")
info_devel_data.index = info_devel_data["f.eid"]


## Step 2. Fit UMAP model
UMAP_full = umap.UMAP(n_neighbors=200, output_metric="euclidean", 
                       min_dist=0.2,
                  random_state=1000).fit(embed_devel_data.iloc[:, 1:])

embed_devel_UMAP = UMAP_full.embedding_


## [SAVE] fitted UMAP model (private)
with open(FITTED_UMAP_PATH, "wb") as f:
    cloudpickle.dump(UMAP_full, f)


## Step 3. Clustering using scanpy library
adata = ad.AnnData(X=embed_devel_data.iloc[:,1:], obs=info_devel_data)
adata.obsm["X_raw"] = adata.X
adata.obsm["umap"] = embed_devel_UMAP
sc.pp.neighbors(adata, n_neighbors=200, use_rep="X_raw", n_pcs=128, random_state=1000)
sc.tl.louvain(adata, resolution=1.2)


## Step 4. Construct PAGA connectivity graphs
sc.tl.paga(adata)
sc.pl.paga(adata, node_size_scale=2.) 


## [SAVE] umap and cluster info
embed_devel_UMAP_cluster = pd.DataFrame(embed_devel_UMAP)
embed_devel_UMAP_cluster['louvain'] = list(adata.obs['louvain'])
embed_devel_UMAP_cluster.to_csv(FULL_DEVEL_UMAP_CLUSTER_PATH)


