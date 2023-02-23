import torch
import torch.nn as nn
import copy
import torch.nn.functional as F
import os
import json
from torch.utils.data import DataLoader
from FLAlgorithms.optimizers.fedoptimizer import MySGD
from FLAlgorithms.users.userbase import User

# Implementation for Per-FedAvg clients
# beta is global level learning rate; learning_rate is personal level learning rate

class UserPerAvg(User):
    def __init__(self, numeric_id, train_data, valid_data, test_data, model, batch_size, learning_rate,beta,lamda,
                 local_epochs, optimizer, total_users , num_users, device=torch.device("cpu"), dataset=None):
        assert dataset is not None
        super().__init__(numeric_id, train_data, valid_data, test_data, model[0], batch_size, learning_rate, beta, lamda,
                         local_epochs, device=device, dataset=dataset)
        self.total_users = total_users
        self.num_users = num_users
        
        self.loss = nn.NLLLoss()

        self.optimizer = MySGD(self.model.parameters(), lr=self.learning_rate)

    def set_grads(self, new_grads):
        if isinstance(new_grads, nn.Parameter):
            for model_grad, new_grad in zip(self.model.parameters(), new_grads):
                model_grad.data = new_grad.data
        elif isinstance(new_grads, list):
            for idx, model_grad in enumerate(self.model.parameters()):
                model_grad.data = new_grads[idx]

    def train(self, batches=None):
        LOSS = 0
        self.model.train()
        if batches is None:
            assert False, "should pass in epochs as arg"
        for batch in range(1, batches + 1):  # local update 
            
            self.model.train()

            temp_model = copy.deepcopy(list(self.model.parameters()))

            #step 1
            X, y = self.get_next_train_batch()
            self.optimizer.zero_grad()
            output = self.model(X)
            loss = self.loss(output, y)
            loss.backward()
            self.optimizer.step()

            #step 2
            X, y = self.get_next_train_batch()
            self.optimizer.zero_grad()
            output = self.model(X)
            loss = self.loss(output, y)
            loss.backward()
            

            # restore the model parameters to the one before first update
            for old_p, new_p in zip(self.model.parameters(), temp_model):
                old_p.data = new_p.data.clone()

            self.optimizer.step(beta = self.beta)

            # clone model to user model 
            # no need to do it for each batch, can move outside the loop, but necessary to do
            self.clone_model_paramenter(self.model.parameters(), self.local_model)
        return LOSS    
        
    def train_n_epochs(self, num_epochs):
        self.model.train()
        assert num_epochs is not None, "should pass in epochs as arg"
        for epoch in range(1, num_epochs + 1):
            self.model.train()

            #TODO the line below was added here by me; the way batching is done in the original code is different; 
            # however, federated averaging the way it was proposed is to do actual epochs as we have it now
            for batch_idx, (X, y) in enumerate(self.trainloader):
                X, y = X.to(self.device), y.to(self.device)
            # X, y = self.get_next_train_batch()
                self.optimizer.zero_grad()
                output = self.model(X)
                loss = self.loss(output, y) 
                loss.backward()
                self.optimizer.step()

