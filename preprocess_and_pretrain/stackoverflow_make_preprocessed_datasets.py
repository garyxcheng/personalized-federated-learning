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
from rnn import *




from tqdm.notebook import tqdm

from FedML.fedml_api.data_preprocessing.stackoverflow_nwp.data_loader_after_pretrain_before_preprocess import load_partition_data_federated_stackoverflow_nwp

tup = load_partition_data_federated_stackoverflow_nwp(None, "./FedML/data/stackoverflow/datasets")

# Update name of model and make appropriate changes once you have the model

dataset = "Stackoverflownwp"

count = 0
a = list(tup[5].keys()).copy()
for i in a:
    if tup[5][i] > 8000:
        del tup[5][i]
        del tup[6][i]
        del tup[7][i]
        count+=1
print(count)

train_frac = 0.8
val_frac = 0.1
test_frac = 0.1

num_users = 300 ## custom


keys = list(tup[5].keys()).copy()

train_data_dict = dict()
val_data_dict = dict()
test_data_dict = dict()
glob_train_data_x = []
glob_train_data_y = []
glob_test_data_x = []
glob_test_data_y = []
glob_val_data_x = []
glob_val_data_y = []

for i in tqdm(np.arange(num_users) + 310):
    user_id = keys[i]
    np.random.seed(int(user_id))
    
    print(user_id)
    train_data_dict[user_id] = {"x" :None, "y" :None}
    val_data_dict[user_id] = {"x" :None, "y" :None}
    test_data_dict[user_id] = {"x" :None, "y" :None}
    
    tot_data_user = len(tup[6][user_id].dataset)
    train_data_user = int(train_frac * tot_data_user)
    val_data_user = int(val_frac * tot_data_user)
    indices = np.arange(tot_data_user)
    np.random.shuffle(indices)
    
    dum_list_x_train = []
    dum_list_y_train = []
    dum_list_x_val = []
    dum_list_y_val = []
    dum_list_x_test = []
    dum_list_y_test = []
    for i in range(tot_data_user):
        if i < train_data_user:
            dum_list_x_train.append(tup[6][user_id].dataset[indices[i]][0])
            dum_list_y_train.append(tup[6][user_id].dataset[indices[i]][1])
        elif i < train_data_user + val_data_user:
            dum_list_x_val.append(tup[6][user_id].dataset[indices[i]][0])
            dum_list_y_val.append(tup[6][user_id].dataset[indices[i]][1])
        else:
            dum_list_x_test.append(tup[6][user_id].dataset[indices[i]][0])
            dum_list_y_test.append(tup[6][user_id].dataset[indices[i]][1])
    
    dum_list_x_train = torch.Tensor(np.stack(dum_list_x_train))
    dum_list_y_train = torch.Tensor(np.stack(dum_list_y_train))
    dum_list_x_val = torch.Tensor(np.stack(dum_list_x_val))
    dum_list_y_val = torch.Tensor(np.stack(dum_list_y_val))
    dum_list_x_test = torch.Tensor(np.stack(dum_list_x_test))
    dum_list_y_test = torch.Tensor(np.stack(dum_list_y_test))
    
    train_data_dict[user_id]["x"] = dum_list_x_train.long()
    train_data_dict[user_id]["y"] = dum_list_y_train.long()

    val_data_dict[user_id]["x"] =  dum_list_x_val.long()
    val_data_dict[user_id]["y"] =  dum_list_y_val.long()

    test_data_dict[user_id]["x"] =  dum_list_x_test.long()
    test_data_dict[user_id]["y"] =  dum_list_y_test.long()
    
    glob_train_data_x.append(dum_list_x_train)
    glob_train_data_y.append(dum_list_y_train)

    glob_val_data_x.append(dum_list_x_val)
    glob_val_data_y.append(dum_list_y_val)

    glob_test_data_x.append(dum_list_x_test)
    glob_test_data_y.append(dum_list_y_test)
    

glob_train_data_x = torch.cat(glob_train_data_x,dim=0).long()
glob_train_data_y = torch.cat(glob_train_data_y,dim=0).long()
glob_val_data_x = torch.cat(glob_val_data_x,dim=0).long()
glob_val_data_y = torch.cat(glob_val_data_y,dim=0).long()
glob_test_data_x = torch.cat(glob_test_data_x,dim=0).long()
glob_test_data_y = torch.cat(glob_test_data_y,dim=0).long()

def preprocess(my_input, model, activation):
    output= model(my_input)
    return activation["fc1"]

preprocessed_data = [list(train_data_dict.keys()).copy(), None, dict(), dict(), dict()]

for user_id in tqdm(preprocessed_data[0]):
    preprocessed_data[2][user_id] = {'x':None,'y':None}
    preprocessed_data[3][user_id] = {'x':None,'y':None}
    preprocessed_data[4][user_id] = {'x':None,'y':None}
    
    print(user_id)
    preprocessed_data[2][user_id]["x"] = preprocess(train_data_dict[user_id]['x'], model, activation)
    preprocessed_data[2][user_id]["y"] = train_data_dict[user_id]['y']

    preprocessed_data[3][user_id]["x"] = preprocess(val_data_dict[user_id]['x'], model, activation)
    preprocessed_data[3][user_id]["y"] = val_data_dict[user_id]['y']
    
    preprocessed_data[4][user_id]["x"] = preprocess(test_data_dict[user_id]['x'], model, activation)
    preprocessed_data[4][user_id]["y"] = test_data_dict[user_id]['y']

try:
    my_file = os.path.join('../personalizedFL','data',dataset,'data','stackoverflownwp_preprocess_data_trial0.p')
    with open(my_file, "wb") as f:
        pickle.dump(preprocessed_data, f)
except Exception:
    import pdb; pdb.set_trace()

train_val_data_dict = train_data_dict.copy()
for i in train_val_data_dict.keys():
    train_val_data_dict[i]['x'] = torch.cat([train_val_data_dict[i]['x'],val_data_dict[i]['x']],dim = 0)
    train_val_data_dict[i]['y'] = torch.cat([train_val_data_dict[i]['y'],val_data_dict[i]['y']],dim = 0)
    
def generate_preprocessed_data(trial_num, train_val_data_dict, test_data_dict, model, activation, train_frac,val_frac):
    preprocessed_data = [list(test_data_dict.keys()), None, dict(), dict(), dict()]
    
    trial_seed = trial_num*1e4
    print("Seed Number "+str(trial_seed))
    
    for user_id in tqdm(preprocessed_data[0]):
        preprocessed_data[2][user_id] = {'x':None,'y':None}
        preprocessed_data[3][user_id] = {'x':None,'y':None}
        preprocessed_data[4][user_id] = {'x':None,'y':None}
        
        train_val_user_size = len(train_val_data_dict[user_id]['y'])
        train_user_size = int(train_val_user_size * (train_frac/(train_frac + val_frac)))
        
        np.random.seed(int(trial_seed + user_id))
        indices = np.arange(train_val_user_size)
        np.random.shuffle(indices)

        print(user_id)
        
        preprocessed_data[2][user_id]["x"] = preprocess(train_val_data_dict[user_id]['x'][indices[:train_user_size]], model, activation)
        preprocessed_data[2][user_id]["y"] = train_val_data_dict[user_id]['y'][indices[:train_user_size]]

        preprocessed_data[3][user_id]["x"] = preprocess(train_val_data_dict[user_id]['x'][indices[train_user_size:]], model, activation)
        preprocessed_data[3][user_id]["y"] = train_val_data_dict[user_id]['y'][indices[train_user_size:]]

        preprocessed_data[4][user_id]["x"] = preprocess(test_data_dict[user_id]['x'], model, activation)
        preprocessed_data[4][user_id]["y"] = test_data_dict[user_id]['y']
    print("Saving file")
    my_file = os.path.join('../personalizedFL','data',dataset,'data','stackoverflownwp_preprocess_data_trial{0}.p'.format(trial_num))
    try:
        with open(my_file, "wb") as f:
            pickle.dump(preprocessed_data, f)
    except Exception:
        import pdb; pdb.set_trace()
    
for trial in np.arange(10)+1:
    generate_preprocessed_data(trial,train_val_data_dict, test_data_dict, model, activation, train_frac,val_frac)