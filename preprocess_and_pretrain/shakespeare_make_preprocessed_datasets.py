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

## Update name of model and make appropriate changes once you have the model
dataset = "Shakespeare"

cpu_device = torch.device('cpu')

activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output[0].detach()
    return hook

model = RNN_OriginalFedAvg().to(cpu_device)
state_dict = torch.load("shakespeare_P1_pretrained_weights/0.01_0.5_shakespeare_rnn_final.pt", map_location=cpu_device) ### update correct path
model.load_state_dict(state_dict, strict=False)
model.lstm.register_forward_hook(get_activation('lstm'))
model.eval()

def preprocess(my_input, model, activation):
    output= model(my_input)
    return activation["lstm"]

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
    

my_file = os.path.join('../personalizedFL','data',dataset,'data','shakespeare_preprocess_data_trial0.p')
with open(my_file, "wb") as f:
    pickle.dump(preprocessed_data, f)
    
train_val_data_dict = train_data_dict.copy()
for i in train_val_data_dict.keys():
    train_val_data_dict[i]['x'] = torch.cat([train_val_data_dict[i]['x'],val_data_dict[i]['x']],dim = 0)
    train_val_data_dict[i]['y'] = torch.cat([train_val_data_dict[i]['y'],val_data_dict[i]['y']],dim = 0)

def generate_preprocessed_data(trial_num, train_val_data_dict, test_data_dict, model, activation, train_frac,val_frac):
    assert trial_num > 0, "if trial num is 0, use the above code"
    preprocessed_data = [list(test_data_dict.keys()), None, dict(), dict(), dict()]
    
    trial_seed = trial_num*1e4
    print("Seed start " + str(trial_seed))
    
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
    my_file = os.path.join('../personalizedFL','data',dataset,'data','shakespeare_preprocess_data_trial{0}.p'.format(trial_num))
    with open(my_file, "wb") as f:
        pickle.dump(preprocessed_data, f)
        
for trial in np.arange(10)+1:
    generate_preprocessed_data(trial,train_val_data_dict, test_data_dict, model, activation, train_frac,val_frac)
