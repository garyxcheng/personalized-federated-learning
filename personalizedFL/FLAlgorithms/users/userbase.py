import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
from torch.utils.data import DataLoader
import numpy as np
import copy

class User:
    """
    Base class for users in federated learning.
    """
    def __init__(self, id, train_data, valid_data, test_data, model, batch_size = 0, learning_rate = 0, beta = 0 , lamda = 0, local_epochs = 0, device=torch.device("cpu"), dataset = None):
        # from fedprox
        if model is None:
            self.model = None
        else:
            self.model = copy.deepcopy(model)
        self.id = id  # integer
        self.train_samples = len(train_data)
        self.valid_samples = len(valid_data)
        self.test_samples = len(test_data)

        if self.train_samples <= 0 or self.valid_samples <=0 or self.test_samples <=0:
            print("0 samples detected in user")
            print(self.id)
            print(self.train_samples)
            print(self.valid_samples)
            print(self.test_samples)

        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.beta = beta
        self.lamda = lamda
        self.local_epochs = local_epochs
        self.device = device

        assert dataset is not None
        self.dataset = dataset

        #TODO: should we shuffle the data?
        #TODO: should we make it a tensordataset?
        self.trainloader = DataLoader(train_data, self.batch_size, shuffle=True, pin_memory=True)
        self.validloader = DataLoader(valid_data, self.batch_size, shuffle=True, pin_memory=True)
        self.testloader =  DataLoader(test_data, self.batch_size, shuffle=True, pin_memory=True)
        if self.model.__class__.__name__ == "Last_Layer_Stackoverflownwp_Net":
            self.testloaderfull = DataLoader(test_data, min(128, self.test_samples), pin_memory=True)#, num_workers=2)
            self.validloaderfull = DataLoader(valid_data, min(128, self.valid_samples), pin_memory=True)#, num_workers=2)
            self.trainloaderfull = DataLoader(train_data, min(128, self.train_samples), pin_memory=True)#, num_workers=2)
        else: 
            self.testloaderfull = DataLoader(test_data, self.test_samples, pin_memory=True)#, num_workers=2)
            self.validloaderfull = DataLoader(valid_data, self.valid_samples, pin_memory=True)#, num_workers=2)
            self.trainloaderfull = DataLoader(train_data, self.train_samples, pin_memory=True)#, num_workers=2)
        self.iter_trainloader = iter(self.trainloader)
        self.iter_validloader = iter(self.validloader)
        self.iter_testloader = iter(self.testloader)

        # those parameters are for persionalized federated learing.
        if self.model is not None:
            self.local_model = copy.deepcopy(list(self.model.parameters()))

            # THIS HAS BEEN MOVED TO userpFedMe
            # self.persionalized_model = copy.deepcopy(list(self.model.parameters()))
            # self.persionalized_model_bar = copy.deepcopy(list(self.model.parameters()))
        
    def set_parameters(self, model):
        for old_param, new_param, local_param in zip(self.model.parameters(), model.parameters(), self.local_model):
            old_param.data = new_param.data.clone()
            local_param.data = new_param.data.clone()
        #self.local_weight_updated = copy.deepcopy(self.optimizer.param_groups[0]['params'])

    def get_parameters(self):
        for param in self.model.parameters():
            param.detach()
        return self.model.parameters()
    
    def clone_model_paramenter(self, param, clone_param):
        for param, clone_param in zip(param, clone_param):
            clone_param.data = param.data.clone()
        return clone_param
    
    def get_updated_parameters(self):
        return self.local_weight_updated
    
    def update_parameters(self, new_params):
        for param , new_param in zip(self.model.parameters(), new_params):
            param.data = new_param.data.clone()

    def get_grads(self):
        # Different compared to earlier code, but doesnt seem like its used anywhere.
        grads = []
        for param in self.model.parameters():
            if param.grad is None:
                grads.append(torch.zeros_like(param.data))
            else:
                grads.append(param.grad.data)
        return grads

    def test(self):
        self.model.eval()
        test_acc = 0
        num_iterations = 0
        with torch.no_grad():
            for x, y in self.testloaderfull:
                num_iterations += 1
                x, y = x.to(self.device), y.to(self.device)
                output = self.model(x)
                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                #print(self.id + ", Test Accuracy:", test_acc / y.shape[0] )
        if num_iterations == 1:
            assert self.test_samples == y.shape[0]
        if self.model.__class__.__name__ == "Last_Layer_CIFAR100_Net" or self.model.__class__.__name__ == "Last_Layer_EMNIST_Net" or self.model.__class__.__name__ =="Last_Layer_Shakespeare_Net":
            assert num_iterations == 1
        if self.model.__class__.__name__ == "Last_Layer_Stackoverflownwp_Net":
            assert y.shape[1] == 20
            return test_acc, self.test_samples * 20 
        elif self.model.__class__.__name__ == "Last_Layer_Shakespeare_Net":
            assert y.shape[1] == 80
            return test_acc, self.test_samples * 80 
        assert len(y.shape)==1 or y.shape[1] == 1
        return test_acc, self.test_samples
    
    def valid(self):
        self.model.eval()
        valid_acc = 0
        num_iterations = 0
        with torch.no_grad():
            for x, y in self.validloaderfull:
                num_iterations += 1
                x, y = x.to(self.device), y.to(self.device)
                output = self.model(x)
                valid_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                #print(self.id + ", Valid Accuracy:", valid_acc / y.shape[0] )
        if num_iterations == 1:
            assert self.valid_samples == y.shape[0]
        if self.model.__class__.__name__ == "Last_Layer_CIFAR100_Net" or self.model.__class__.__name__ == "Last_Layer_EMNIST_Net" or self.model.__class__.__name__ =="Last_Layer_Shakespeare_Net":
            assert num_iterations == 1
        if self.model.__class__.__name__ == "Last_Layer_Stackoverflownwp_Net":
            assert y.shape[1] == 20
            return valid_acc, self.valid_samples * 20 
        elif self.model.__class__.__name__ == "Last_Layer_Shakespeare_Net":
            assert y.shape[1] == 80
            return valid_acc, self.valid_samples * 80 
        assert len(y.shape)==1 or y.shape[1] == 1
        return valid_acc, self.valid_samples

    def train_error_and_loss(self):
        self.model.eval()
        train_acc = 0
        loss = 0
        num_iterations = 0
        with torch.no_grad():
            for x, y in self.trainloaderfull:
                num_iterations += 1
                x, y = x.to(self.device), y.to(self.device)
                output = self.model(x)
                train_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                loss += self.loss(output, y)
                #print(self.id + ", Train Accuracy:", train_acc)
                #print(self.id + ", Train Loss:", loss)
        if num_iterations == 1:
            assert self.train_samples == y.shape[0]
        if self.model.__class__.__name__ == "Last_Layer_CIFAR100_Net" or self.model.__class__.__name__ == "Last_Layer_EMNIST_Net" or self.model.__class__.__name__ =="Last_Layer_Shakespeare_Net":
            assert num_iterations == 1
        if self.model.__class__.__name__ == "Last_Layer_Stackoverflownwp_Net":
            assert y.shape[1] == 20
            return train_acc, loss, self.train_samples* 20 
        elif self.model.__class__.__name__ == "Last_Layer_Shakespeare_Net":
            assert y.shape[1] == 80
            return train_acc, loss, self.train_samples * 80 
        assert len(y.shape)==1 or y.shape[1] == 1
        return train_acc, loss, self.train_samples
    
    def test_persionalized_model(self):
        self.model.eval()
        test_acc = 0
        num_iterations = 0
        self.update_parameters(self.persionalized_model_bar)
        # think about any possible memory overflow issues due to update_params
        with torch.no_grad():
            for x, y in self.testloaderfull:
                num_iterations += 1
                x, y = x.to(self.device), y.to(self.device)
                output = self.model(x)
                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                #print(self.id + ", Test Accuracy:", test_acc / y.shape[0] )
        self.update_parameters(self.local_model)
        if num_iterations == 1:
            assert self.test_samples == y.shape[0]
        if self.model.__class__.__name__ == "Last_Layer_CIFAR100_Net" or self.model.__class__.__name__ == "Last_Layer_EMNIST_Net" or self.model.__class__.__name__ =="Last_Layer_Shakespeare_Net":
            assert num_iterations == 1
        if self.model.__class__.__name__ == "Last_Layer_Stackoverflownwp_Net":
            assert y.shape[1] == 20
            return test_acc, self.test_samples * 20 
        elif self.model.__class__.__name__ == "Last_Layer_Shakespeare_Net":
            assert y.shape[1] == 80
            return test_acc, self.test_samples * 80 
        assert len(y.shape)==1 or y.shape[1] == 1
        return test_acc, self.test_samples

    def valid_persionalized_model(self):
        self.model.eval()
        valid_acc = 0
        num_iterations = 0
        self.update_parameters(self.persionalized_model_bar)
        # think about any possible memory overflow issues due to update_params
        with torch.no_grad():
            for x, y in self.validloaderfull:
                num_iterations += 1
                x, y = x.to(self.device), y.to(self.device)
                output = self.model(x)
                valid_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                #print(self.id + ", Valid Accuracy:", valid_acc / y.shape[0] )
        self.update_parameters(self.local_model)
        if num_iterations == 1:
            assert self.valid_samples == y.shape[0]
        if self.model.__class__.__name__ == "Last_Layer_CIFAR100_Net" or self.model.__class__.__name__ == "Last_Layer_EMNIST_Net" or self.model.__class__.__name__ =="Last_Layer_Shakespeare_Net":
            assert num_iterations == 1
        if self.model.__class__.__name__ == "Last_Layer_Stackoverflownwp_Net":
            assert y.shape[1] == 20
            return valid_acc, self.valid_samples * 20 
        elif self.model.__class__.__name__ == "Last_Layer_Shakespeare_Net":
            assert y.shape[1] == 80
            return valid_acc, self.valid_samples * 80 
        assert len(y.shape)==1 or y.shape[1] == 1
        return valid_acc, self.valid_samples
    
    def train_error_and_loss_persionalized_model(self):
        self.model.eval()
        train_acc = 0
        loss = 0
        num_iterations = 0
        self.update_parameters(self.persionalized_model_bar)
        with torch.no_grad():
            for x, y in self.trainloaderfull:
                num_iterations += 1
                x, y = x.to(self.device), y.to(self.device)
                output = self.model(x)
                train_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                loss += self.loss(output, y)
                #print(self.id + ", Train Accuracy:", train_acc)
                #print(self.id + ", Train Loss:", loss)
        self.update_parameters(self.local_model)
        if num_iterations == 1:
            assert self.train_samples == y.shape[0]
        if self.model.__class__.__name__ == "Last_Layer_CIFAR100_Net" or self.model.__class__.__name__ == "Last_Layer_EMNIST_Net" or self.model.__class__.__name__ =="Last_Layer_Shakespeare_Net":
            assert num_iterations == 1
        if self.model.__class__.__name__ == "Last_Layer_Stackoverflownwp_Net":
            assert y.shape[1] == 20
            return train_acc, loss, self.train_samples* 20 
        elif self.model.__class__.__name__ == "Last_Layer_Shakespeare_Net":
            assert y.shape[1] == 80
            return train_acc, loss, self.train_samples * 80 
        assert len(y.shape)==1 or y.shape[1] == 1
        return train_acc, loss , self.train_samples
    
    def get_next_train_batch(self):
        try:
            # Samples a new batch for persionalizing
            (X, y) = next(self.iter_trainloader)
            X, y = X.to(self.device), y.to(self.device)
        except StopIteration:
            # restart the generator if the previous generator is exhausted.
            self.iter_trainloader = iter(self.trainloader)
            (X, y) = next(self.iter_trainloader)
            X, y = X.to(self.device), y.to(self.device)
        return (X, y)
    
    def get_next_test_batch(self):
        try:
            # Samples a new batch for persionalizing
            (X, y) = next(self.iter_testloader)
            X, y = X.to(self.device), y.to(self.device)
        except StopIteration:
            # restart the generator if the previous generator is exhausted.
            self.iter_testloader = iter(self.testloader)
            (X, y) = next(self.iter_testloader)
            X, y = X.to(self.device), y.to(self.device)
        return (X, y)

    def save_model(self):
        model_path = os.path.join("models", self.dataset)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(self.model, os.path.join(model_path, "user_" + self.id + ".pt"))

    def load_model(self):
        model_path = os.path.join("models", self.dataset)
        self.model = torch.load(os.path.join(model_path, "server" + ".pt"))
    
    @staticmethod
    def model_exists():
        return os.path.exists(os.path.join("models", "server" + ".pt"))