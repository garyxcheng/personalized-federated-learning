import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import copy
import json
from torch.utils.data import DataLoader
from FLAlgorithms.users.userbase import User
# from FLAlgorithms.users.useravg import UserAVG

# Implementation for UserPerRidgeEpoch clients
#this is basically a copy of UserAVG but with custom train_n_epochs which trains with a ridge penalty
# beta is global level learning rate; learning_rate is personal level learning rate

class UserPerRidgeEpochBatch(User):
    #TODO add ridge penalty
    def __init__(self, numeric_id, train_data, valid_data, test_data, model, batch_size, learning_rate, beta, lamda,
                 local_epochs, optimizer, device=torch.device("cpu"), dataset=None, user_ridge_penalty = None):
        assert user_ridge_penalty is not None
        assert dataset is not None
        self.user_ridge_penalty = user_ridge_penalty
        super().__init__(numeric_id, train_data, valid_data, test_data, model[0], batch_size, learning_rate, beta, lamda,
                         local_epochs, device=device, dataset=dataset)

        self.loss = nn.NLLLoss()

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.beta)
        self.personal_optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)

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
        for batch in range(1, batches + 1):
            self.model.train()
            # for batch_idx, (X, y) in enumerate(self.trainloader):
            X, y = self.get_next_train_batch()
            X, y = X.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(X)
            loss = self.loss(output, y) 
            loss.backward()
            self.optimizer.step()
        self.clone_model_paramenter(self.model.parameters(), self.local_model)
        return LOSS

    #this function is different from other peravg type algorithms as we use a self.personal_optimizer here
    def train_n_epochs(self, num_epochs):
        self.model.train()
        assert num_epochs is not None, "should pass in epochs as arg"
        deep_copy_params = copy.deepcopy(list(self.model.parameters()))
        for epoch in range(1, num_epochs + 1):
            self.model.train()

            for batch_idx, (X, y) in enumerate(self.trainloader):
                X, y = X.to(self.device), y.to(self.device)
                self.personal_optimizer.zero_grad()
                output = self.model(X)
                loss = self.loss(output, y) 
                l2_reg = torch.tensor(0.).to(self.device)
                for idx, param in enumerate(self.model.parameters()):
                    if param.requires_grad:
                        l2_reg += torch.square(torch.norm(param- deep_copy_params[idx]))
                loss += self.user_ridge_penalty * l2_reg

                loss.backward()
                self.personal_optimizer.step()

