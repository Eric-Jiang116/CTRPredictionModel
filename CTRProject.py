
import os
import pandas as pd
import numpy as np
import torch.nn as nn
import torch

path = "/Users/ericjiang/.cache/kagglehub/datasets/datuman/criteo-ad-click-limited-1m/versions/4"
file_name = os.listdir(path)[0] 
file_path = os.path.join(path, file_name)

Data = pd.read_csv(file_path)
intCols = [f"intCol_{i}" for i in range(13)] # 12 Integer Columns
CatCols = [f"catCol_{i}" for i in range(26)] # 26 cat Columns

Data[intCols] = Data[intCols].fillna(0).astype('float32')
Data[CatCols] = Data[CatCols].fillna("missing").astype(str)

for cat in CatCols:
    Data[cat], _ = pd.factorize(Data[cat])
for col in intCols:
    Data[col] = np.log1p(Data[col].clip(lower=0))

print(Data.head(5))
""" vocabsizes = []
for cat in CatCols:
    vocabsizes.append(Data[cat].nunique())
Data.to_csv("criteo_processed.csv", index=False)
import json
with open("vocab_sizes.json", "w") as f:
    json.dump(vocabsizes, f)

print("CSV isss ready.") """