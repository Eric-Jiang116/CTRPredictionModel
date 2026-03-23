import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import json
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("🚀 Using GPU: Apple Metal Performance Shaders")

with open("vocab_sizes.json", "r") as f:
    catunique = json.load(f)

Data = pd.read_csv("criteo_processed.csv")

intCols = [f"intCol_{i}" for i in range(13)] # 13 Integer Columns
CatCols = [f"catCol_{i}" for i in range(26)] # 26 cat Columns
x_cat = Data[CatCols].copy()
x_num = Data[intCols].copy()
y_target = Data["target"].copy()

xcat_train, xcattemp, xnum_train, xnumtemp, y_train, ytemp = train_test_split(x_cat, x_num, y_target, test_size= 0.2)

xcat_val, xcat_test, xnum_val, xnum_test, y_val, y_test = train_test_split(xcattemp, xnumtemp, ytemp, test_size= 0.5)

#normalize
xnum_train_mean = xnum_train.mean()
xnum_train_std = xnum_train.std()

xnum_train = (xnum_train - xnum_train_mean) / xnum_train_std
xnum_val = (xnum_val - xnum_train_mean) / xnum_train_std
xnum_test = (xnum_test - xnum_train_mean) / xnum_train_std

xnum_train = torch.tensor(xnum_train.to_numpy(), dtype=torch.float32)
xnum_val = torch.tensor(xnum_val.to_numpy(), dtype=torch.float32)
xnum_test = torch.tensor(xnum_test.to_numpy(), dtype=torch.float32)

xcat_train = torch.tensor(xcat_train.to_numpy(), dtype=torch.long)
xcat_val = torch.tensor(xcat_val.to_numpy(), dtype=torch.long)
xcat_test = torch.tensor(xcat_test.to_numpy(), dtype=torch.long)

y_train = torch.tensor(y_train.to_numpy(), dtype=torch.float32)
y_val = torch.tensor(y_val.to_numpy(), dtype=torch.float32)
y_test = torch.tensor(y_test.to_numpy(), dtype=torch.float32)
    
trainDataset = TensorDataset(xnum_train, xcat_train, y_train)
validationDataset = TensorDataset(xnum_val, xcat_val, y_val)
testDataset = TensorDataset(xnum_test, xcat_test, y_test)

train_loader = DataLoader(trainDataset, batch_size=2048, shuffle=True)
val_loader = DataLoader(validationDataset, batch_size=4096, shuffle=False)
test_loader = DataLoader(testDataset, batch_size=4096, shuffle = False)

class DeepFM(nn.Module):
    def __init__(self, cat_unique_vals , num_int_features, embdim=16): #cat_unique_vals shape = (26)
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(uniquevals, embdim) for uniquevals in cat_unique_vals])
        self.categoriesamt = len(cat_unique_vals)

        #firstOrderLinear Part
        self.linear_cat = nn.ModuleList([
             nn.Embedding(uniquevals, 1) for uniquevals in cat_unique_vals
            ])
        self.linear_num = nn.Linear(num_int_features, 1)
        self.bias = nn.Parameter(torch.zeros(1))

        inputsize = len(cat_unique_vals) * embdim + num_int_features
        self.deepLayer = nn.Sequential(nn.Linear(inputsize, 512),
            nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, x_int, x_cat):
        #x_int = numerical dataframe shape (128, 13)
        #x_cat = already translated categorical dataframe
        embedded_vectors = torch.stack([embed(x_cat[:,i]) for i, embed in enumerate(self.embeddings)], dim = 1) 
        #shape = (Batch of people ,26, embdimension(16))
        #remember x_cat height is the same among all.
        square_of_sum = (torch.sum(embedded_vectors, dim=1)) ** 2
        sum_of_square = torch.sum(embedded_vectors ** 2, dim=1)
    
        # We sum across the 16 dims to get one number per row
        fm_score = 0.5 * torch.sum(square_of_sum - sum_of_square, dim=1, keepdim=True)
        #first order linear
        linear_cat_terms = [emb(x_cat[:, i]) for i, emb in enumerate(self.linear_cat)]
        linear_cat_score = torch.sum(torch.cat(linear_cat_terms, dim=1), dim=1, keepdim=True)
        linear_num_score = self.linear_num(x_int)

        # Flatten the 26 vectors into one long 416-dim vector, = 26 x 16
        cat_flat = embedded_vectors.view(embedded_vectors.size(0), -1) #shape(batch, 416)
        # Concatenate with the 13 integers
        deep_input = torch.cat([cat_flat, x_int], dim=1) #shape (batch, 416 + 13 = 429)
        deep_score = self.deepLayer(deep_input)
        # We add them together and squash into a 0 to 1 probability
        final_output = self.bias + linear_cat_score + linear_num_score+ fm_score + deep_score
        return final_output

model = DeepFM(catunique, 13, 32)

model.to(device)

lossfn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 2e-4)
epochs = 20
#train
for epoch in range(epochs):
    model.train()
    totalTrainLoss = 0
    for numBatch, catBatch, targetbatch in train_loader:
        numBatch = numBatch.to(device)
        catBatch = catBatch.to(device)
        targetbatch = targetbatch.to(device).view(-1, 1)
        optimizer.zero_grad()
        pred = model(numBatch, catBatch)
        loss = lossfn(pred, targetbatch.view(-1,1))
        totalTrainLoss += loss.item()
        loss.backward()
        optimizer.step()
    model.eval()
    totalValLoss = 0
    val_preds = []
    val_targets = []
    with torch.no_grad():
        for numBatch, catBatch, targetbatch in val_loader:
            numBatch = numBatch.to(device)
            catBatch = catBatch.to(device)
            targetbatch = targetbatch.to(device).view(-1, 1)

            valPred = model(numBatch, catBatch)
            valLoss = lossfn(valPred, targetbatch.view(-1,1))
            totalValLoss += valLoss.item()
            #ROCAUC stuff
            probs = torch.sigmoid(valPred).cpu().numpy()
            val_preds.extend(probs)
            val_targets.extend(targetbatch.cpu().numpy())
    epoch_auc = roc_auc_score(val_targets, val_preds)
    print(f"Val AUC: {epoch_auc:.4f}")
    amtBatches = len(val_loader)
    amtTrainBatches = len(train_loader)
    avgValLoss = totalValLoss / amtBatches
    avgTrainLoss = totalTrainLoss / amtTrainBatches
    print(f"Epoch {epoch+1} | Train Loss: {avgTrainLoss} | Val Loss: {avgValLoss}")

