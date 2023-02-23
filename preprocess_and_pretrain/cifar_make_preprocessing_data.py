from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import os
import json
import numpy as np
import pickle
import torchvision.models as models
import time

from tqdm.notebook import tqdm
from FedML.fedml_api.data_preprocessing.fed_cifar100.data_loader import load_partition_data_federated_cifar100

tup = load_partition_data_federated_cifar100(None, "./FedML/data/fed_cifar100/datasets")
# tup[6] and tup[7] are dictionaries with train and test data loaders of users respectively 

train_frac = 0.8
val_frac = 0.1
test_frac = 0.1

num_users = 600 ## custom

train_data_dict = dict()
test_data_dict = dict()
val_data_dict = dict()

glob_train_data_x = []
glob_train_data_y = []
glob_test_data_x = []
glob_test_data_y = []
glob_val_data_x = []
glob_val_data_y = []

for user_id in tqdm(range(num_users)):
    np.random.seed(int(user_id))
    
    train_data_dict[user_id] = {"x" :None, "y" :None}
    val_data_dict[user_id] = {"x" :None, "y" :None}
    test_data_dict[user_id] = {"x" :None, "y" :None}
    
    if user_id < 500:
        # always length 100 for CIFAR
        tot_data_user = len(tup[6][user_id].dataset[:][0])
        train_data_user = int(train_frac * tot_data_user)
        val_data_user = int(val_frac * tot_data_user)
        indices = np.arange(tot_data_user)
        np.random.shuffle(indices)
        
        glob_train_data_x.append(tup[6][user_id].dataset[indices[:train_data_user]][0])
        glob_train_data_y.append(tup[6][user_id].dataset[indices[:train_data_user]][1])
        
        train_data_dict[user_id]["x"] = tup[6][user_id].dataset[indices[:train_data_user]][0]
        train_data_dict[user_id]["y"] = tup[6][user_id].dataset[indices[:train_data_user]][1]


        glob_val_data_x.append(tup[6][user_id].dataset[indices[train_data_user:train_data_user + val_data_user]][0])
        glob_val_data_y.append(tup[6][user_id].dataset[indices[train_data_user:train_data_user + val_data_user]][1])
        
        val_data_dict[user_id]["x"] =  tup[6][user_id].dataset[indices[train_data_user:train_data_user + val_data_user]][0]
        val_data_dict[user_id]["y"] =  tup[6][user_id].dataset[indices[train_data_user:train_data_user + val_data_user]][1]
        
        glob_test_data_x.append(tup[6][user_id].dataset[indices[train_data_user + val_data_user:]][0])
        glob_test_data_y.append(tup[6][user_id].dataset[indices[train_data_user + val_data_user:]][1])
        
        test_data_dict[user_id]["x"] =  tup[6][user_id].dataset[indices[train_data_user + val_data_user:]][0]
        test_data_dict[user_id]["y"] =  tup[6][user_id].dataset[indices[train_data_user + val_data_user:]][1]
        
    else:
        id_new = user_id - 500
        
        tot_data_user = len(tup[6][id_new].dataset[:][0])
        train_data_user = int(train_frac * tot_data_user)
        val_data_user = int(val_frac * tot_data_user)
        indices = np.arange(tot_data_user)
        np.random.shuffle(indices)
        
        glob_train_data_x.append(tup[7][id_new].dataset[indices[:train_data_user]][0])
        glob_train_data_y.append(tup[7][id_new].dataset[indices[:train_data_user]][1])
        
        train_data_dict[user_id]["x"] = tup[7][id_new].dataset[indices[:train_data_user]][0]
        train_data_dict[user_id]["y"] = tup[7][id_new].dataset[indices[:train_data_user]][1]


        glob_val_data_x.append(tup[7][id_new].dataset[indices[train_data_user:train_data_user + val_data_user]][0])
        glob_val_data_y.append(tup[7][id_new].dataset[indices[train_data_user:train_data_user + val_data_user]][1])
        
        val_data_dict[user_id]["x"] =  tup[7][id_new].dataset[indices[train_data_user:train_data_user + val_data_user]][0]
        val_data_dict[user_id]["y"] =  tup[7][id_new].dataset[indices[train_data_user:train_data_user + val_data_user]][1]
        
        glob_test_data_x.append(tup[7][id_new].dataset[indices[train_data_user + val_data_user:]][0])
        glob_test_data_y.append(tup[7][id_new].dataset[indices[train_data_user + val_data_user:]][1])
        
        test_data_dict[user_id]["x"] =  tup[7][id_new].dataset[indices[train_data_user + val_data_user:]][0]
        test_data_dict[user_id]["y"] =  tup[7][id_new].dataset[indices[train_data_user + val_data_user:]][1]
        
# global train data as tensors

glob_train_data_x = torch.cat(glob_train_data_x,dim=0)
glob_train_data_y = torch.cat(glob_train_data_y,dim=0)
glob_val_data_x = torch.cat(glob_val_data_x,dim=0)
glob_val_data_y = torch.cat(glob_val_data_y,dim=0)
glob_test_data_x = torch.cat(glob_test_data_x,dim=0)
glob_test_data_y = torch.cat(glob_test_data_y,dim=0)

#convert the x data to float and divide by 255
glob_train_data_x = torch.Tensor.float(glob_train_data_x)/255.0
glob_test_data_x = torch.Tensor.float(glob_test_data_x)/255.0
glob_val_data_x = torch.Tensor.float(glob_val_data_x)/255.0

torch.save((glob_train_data_x, glob_train_data_y, glob_val_data_x, glob_val_data_y, glob_test_data_x, glob_test_data_y), 'cifar100_data_federated_split.pt')