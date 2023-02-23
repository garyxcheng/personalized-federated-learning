from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from rnn import *
import os
import json
import numpy as np
import pickle

from tqdm import tqdm_notebook as tqdm

from FedML.fedml_api.data_preprocessing.fed_shakespeare.data_loader import load_partition_data_federated_shakespeare

tup = load_partition_data_federated_shakespeare(None, "./FedML/data/fed_shakespeare/datasets")

count = 0
a = list(tup[5].keys()).copy()
for i in a:
    if tup[5][i] < 2:
        del tup[5][i]
        del tup[6][i]
        del tup[7][i]
        count+=1
print(count)

train_frac = 0.8
val_frac = 0.1
test_frac = 0.1
num_users= len(tup[5])

train_data_dict = dict()
test_data_dict = dict()
val_data_dict = dict()

glob_train_data_x = []
glob_train_data_y = []
glob_test_data_x = []
glob_test_data_y = []
glob_val_data_x = []
glob_val_data_y = []

for user_id in tqdm(list(tup[6].keys())):
    np.random.seed(int(user_id))
    
    train_data_dict[user_id] = {"x" :None, "y" :None}
    val_data_dict[user_id] = {"x" :None, "y" :None}
    test_data_dict[user_id] = {"x" :None, "y" :None}

    train_val_data_user = len(tup[6][user_id].dataset[:][0])
    train_data_user = int(train_frac * train_val_data_user/(train_frac + val_frac))
    indices = np.arange(train_val_data_user)
    np.random.shuffle(indices)

    glob_train_data_x.append(tup[6][user_id].dataset[indices[:train_data_user]][0])
    glob_train_data_y.append(tup[6][user_id].dataset[indices[:train_data_user]][1])

    train_data_dict[user_id]["x"] = tup[6][user_id].dataset[indices[:train_data_user]][0]
    train_data_dict[user_id]["y"] = tup[6][user_id].dataset[indices[:train_data_user]][1]


    glob_val_data_x.append(tup[6][user_id].dataset[indices[train_data_user:]][0])
    glob_val_data_y.append(tup[6][user_id].dataset[indices[train_data_user:]][1])

    val_data_dict[user_id]["x"] =  tup[6][user_id].dataset[indices[train_data_user:]][0]
    val_data_dict[user_id]["y"] =  tup[6][user_id].dataset[indices[train_data_user:]][1]

    glob_test_data_x.append(tup[7][user_id].dataset[:][0])
    glob_test_data_y.append(tup[7][user_id].dataset[:][1])

    test_data_dict[user_id]["x"] =  tup[7][user_id].dataset[:][0]
    test_data_dict[user_id]["y"] =  tup[7][user_id].dataset[:][1]
    
# global train data as tensors

glob_train_data_x = torch.cat(glob_train_data_x,dim=0)
glob_train_data_y = torch.cat(glob_train_data_y,dim=0)
glob_val_data_x = torch.cat(glob_val_data_x,dim=0)
glob_val_data_y = torch.cat(glob_val_data_y,dim=0)
glob_test_data_x = torch.cat(glob_test_data_x,dim=0)
glob_test_data_y = torch.cat(glob_test_data_y,dim=0)

torch.save((glob_train_data_x, glob_train_data_y, glob_val_data_x, glob_val_data_y, glob_test_data_x, glob_test_data_y), 'shakespeare_data_federated_split.pt')