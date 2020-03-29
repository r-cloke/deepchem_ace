#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 14:45:01 2020

@author: ryan
"""


from rdkit import Chem
from rdkit.Chem import Draw
import pandas as pd
from rdkit.Chem import PandasTools
from rdkit.Chem import Descriptors
from rdkit.Chem import rdmolops
import seaborn as sns
import deepchem as dc
from deepchem.models import GraphConvModel
import numpy as np
import sys
import pandas as pd
import deepchem as dc
import pandas as pd
from rdkit.Chem import PandasTools, Draw
from rdkit import DataStructs
from rdkit.ML.Cluster import Butina
from rdkit.Chem import rdMolDescriptors as rdmd
import seaborn as sns

def make_input():
    active_df = pd.read_csv("actives_final.ism", header=None, sep=" ")
    active_rows, active_cols = active_df.shape
    active_df.columns = ["SMILES","ID","ChEMBL_ID"]
    active_df["label"] = ["Active"]*active_rows
    PandasTools.AddMoleculeColumnToFrame(active_df, "SMILES", "MOL")
    
    decoy_df = pd.read_csv("decoys_final.ism", header=None, sep=" ")
    decoy_rows, decoy_cols = decoy_df.shape
    decoy_df.columns = ["SMILES","ID"]
    decoy_df["label"] = ["Decoy"]*decoy_rows
    PandasTools.AddMoleculeColumnToFrame(decoy_df, "SMILES", "MOL")
    
    active_df["is_active"] = [1]*active_df.shape[0]
    decoy_df["is_active"] = [0]*decoy_df.shape[0]
    combined_df = active_df.append(decoy_df)[["SMILES","ID","is_active"]]
    
    
    combined_df.to_csv("dude_ace.csv", index=False)

def generate_graph_conv_model():
    batch_size = 128
    model = GraphConvModel(1, batch_size=batch_size,mode="classification",model_dir="/tmp/mk01/model_dir")
    
    dataset_file = "dude_ace.csv"
    tasks = ["is_active"]
    featurizer = dc.feat.ConvMolFeaturizer()
    loader = dc.data.CSVLoader(tasks=tasks,smiles_field="SMILES",featurizer=featurizer)
    dataset = loader.featurize(dataset_file, shard_size=8192)

    splitter = dc.splits.RandomSplitter()

    metrics = [
        dc.metrics.Metric(dc.metrics.matthews_corrcoef, np.mean, mode="classification")]

    training_score_list = []
    validation_score_list = []
    transformers = []

    model.fit(dataset)
    print(model.evaluate(dataset, metrics))
    return model

#model = generate_graph_conv_model()
model = GraphConvModel(1, batch_size=128,mode="classification",model_dir="/tmp/mk01/model_dir")
model.restore()
#make predictions
featurizer = dc.feat.ConvMolFeaturizer()
df = pd.read_csv("zinc_100k.txt",sep=" ",delimiter=' ',header=None)
df.columns=["SMILES","Name"]

rows,cols = df.shape
df["Val"] = [0] * rows #just add add a dummy column to keep the featurizer happy
infile_name = "zinc_filtered.csv"
df.to_csv(infile_name,index=False)
loader = dc.data.CSVLoader(tasks=['Val'], smiles_field="SMILES", featurizer=featurizer)
dataset = loader.featurize(infile_name, shard_size=8192)
pred = model.predict(dataset)
pred_df = pd.DataFrame([x.flatten() for x in pred],columns=["Neg","Pos"])
sns.distplot(pred_df.Pos,rug=True)
combo_df = df.join(pred_df,how="outer")
combo_df.sort_values("Pos",inplace=True,ascending=False)
PandasTools.AddMoleculeColumnToFrame(combo_df,"SMILES","Mol")

combo_df.to_csv('zinc_output_predictions.csv',sep=',')