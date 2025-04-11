import os, sys
import dill
import pandas as pd
import numpy as np

import faiss


######## data paths  
FULL_DEVEL_EMBED_PATH = "private/devel_imputed_embeddings_full.csv"
FULL_DEVEL_UMAP_CLUSTER_PATH = "private/devel_full_umap_cluster.csv"
FITTED_UMAP_PATH = "private/fitted_UMAP_model.pkl"
TEST_EMBED_PATH = "model/pretrained_model_with_dae/test_imputed_embeddings.csv"
TEST_UMAP_CLUSTER_PATH = "model/pretrained_model_with_dae/test_umap_cluster.csv"

with open(FITTED_UMAP_PATH, "rb") as f:
    UMAP_full = dill.load(f)


## Step 1. Load test embeddings and trasnform to UMAP space 
embed_test_data = pd.read_csv(TEST_EMBED_PATH, sep=",")
embed_test_data = embed_test_data[[*["MEAN"+str(i) for i in range(1, 129)]]]

test_UMAP = UMAP_full.transform(embed_test_data.to_numpy())


## Step 2. Estimate clusters of test data using faiss package
embed_devel_data = pd.read_csv(FULL_DEVEL_EMBED_PATH)
embed_devel_data = embed_devel_data[["f.eid", *["MEAN"+str(i) for i in range(1, 129)]]]
embed_devel_data.index = embed_devel_data["f.eid"]

embed_devel_UMAP_cluster = pd.read_csv(FULL_DEVEL_UMAP_CLUSTER_PATH)

## Faiss K-NN class from https://gist.github.com/j-adamczyk/74ee808ffd53cd8545a49f185a908584
class FaissKNeighbors:
    def __init__(self, k=5):
        self.index = None
        self.y = None
        self.k = k

    def fit(self, X, y):
        self.index = faiss.IndexFlatL2(X.shape[1])
        self.index.add(X.astype(np.float32))
        self.y = y

    def predict(self, X):
        distances, indices = self.index.search(X.astype(np.float32), k=self.k)
        votes = self.y[indices]
        predictions = np.array([np.argmax(np.bincount(x)) for x in votes])
        return predictions


fknn = FaissKNeighbors(k=200)
fknn.fit(embed_devel_data.iloc[:, 1:129], embed_devel_UMAP_cluster['louvain'].to_numpy())
test_prdicted_clusters = fknn.predict(embed_test_data)


## [SAVE] umap and cluster info
test_UMAP_cluster = pd.DataFrame(test_UMAP)
test_UMAP_cluster['louvain'] = test_prdicted_clusters
test_UMAP_cluster.to_csv(TEST_UMAP_CLUSTER_PATH)



