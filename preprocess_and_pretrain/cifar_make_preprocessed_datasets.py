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

dataset = "CIFAR100"

cpu_device = torch.device('cpu')

activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

from cnn import *

model = resnet18().to(cpu_device)
state_dict = torch.load("./cifar100_P1_pretrained_weights/0.1_cifar100_cnn_final.pt", map_location=cpu_device)
model.load_state_dict(state_dict, strict=False)
model.intermediate_layer.register_forward_hook(get_activation('intermediate_layer'))
model.eval()

mean = torch.mean(glob_train_data_x, axis=tuple([0,2,3]))
print(mean)
std = torch.std(glob_train_data_x, axis=tuple([0,2,3]))
print(std)

def preprocess(my_input, model, activation):
    intermediate_input = torch.Tensor.float(my_input)/255.0
    #normalize input
    intermediate_input[:,0,:,:] = (intermediate_input[:,0,:,:] - mean[0])/ std[0]
    intermediate_input[:,1,:,:] = (intermediate_input[:,1,:,:] - mean[1])/ std[1]
    intermediate_input[:,2,:,:] = (intermediate_input[:,2,:,:] - mean[2])/ std[2]
    output= model(intermediate_input)
    return activation["intermediate_layer"].view(activation["intermediate_layer"].size(0), -1)

preprocessed_data = [list(train_data_dict.keys()).copy(), None, dict(), dict(), dict()]

########## Generate Trial 0 data ##################

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

my_file = os.path.join('../personalizedFL','data',dataset,'data','cifar100_cutoutresnet18preprocess_data_trial0.p')
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
    my_file = os.path.join('../personalizedFL','data',dataset,'data','cifar100_cutoutresnet18preprocess_data_trial{0}.p'.format(trial_num))
    with open(my_file, "wb") as f:
        pickle.dump(preprocessed_data, f)

for trial in np.arange(10)+1:
    generate_preprocessed_data(trial,train_val_data_dict, test_data_dict, model, activation, train_frac,val_frac)