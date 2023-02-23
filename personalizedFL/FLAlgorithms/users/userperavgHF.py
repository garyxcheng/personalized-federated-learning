import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
from torch.utils.data import DataLoader
from FLAlgorithms.optimizers.fedoptimizer import MySGD
from FLAlgorithms.users.userbase import User
import copy

# Implementation for Per-FedAvg clients

class UserPerAvgHF(User):
    def __init__(self, numeric_id, train_data, valid_data, test_data, model, batch_size, learning_rate, beta,lamda,
                 local_epochs, optimizer, total_users , num_users, beta_HF, delta_HF, device=torch.device("cpu"), dataset=None):
        assert dataset is not None

        ##### ALPHA in HF algorithm is LR ########
        ##### DELTA in HF algorithm is delta_HF #####
        ##### Beta in HF algorithm is beta_HF #####

        super().__init__(numeric_id, train_data, valid_data, test_data, model[0], batch_size, learning_rate, beta, lamda,
                         local_epochs, device=device, dataset=dataset)
        self.total_users = total_users
        self.num_users = num_users
        self.beta_HF = beta_HF
        self.delta_HF = delta_HF
        
        self.loss = nn.NLLLoss()

        self.optimizer = MySGD(self.model.parameters(), lr=self.learning_rate)

    def set_grads(self, new_grads):
        if isinstance(new_grads, nn.Parameter):
            for model_grad, new_grad in zip(self.model.parameters(), new_grads):
                model_grad.data = new_grad.data
        elif isinstance(new_grads, list):
            for idx, model_grad in enumerate(self.model.parameters()):
                model_grad.data = new_grads[idx]

    def train(self, epochs=None):
        LOSS = 0
        self.model.train()
        if epochs is None:
            assert False, "should pass in epochs as arg"
        for epoch in range(1, epochs + 1):  # local update 
            
            self.model.train()

            ## w ##
            model_at_start = copy.deepcopy(list(self.model.parameters()))
            #step 1 (first grad step) store this
            X, y = self.get_next_train_batch()
            self.optimizer.zero_grad()
            output = self.model(X)
            loss = self.loss(output, y)
            loss.backward()
            self.optimizer.step() # replace with appropriate function

            # Gradient evaluated at first gradient
            X_p, y_p = self.get_next_train_batch()
            self.optimizer.zero_grad()
            output = self.model(X_p)
            loss = self.loss(output, y_p)
            #loss.backward()
            ## \grad f(w - \alpha \grad f(w)) ##
            ## stored in model_1_step[i].grad ##
            model_1_step_grads = torch.autograd.grad(loss, self.model.parameters())
            ## w - \alpha \grad f(w) ###
            # model_1_step = copy.deepcopy(list(self.model.parameters()))

            new_params_plus_delta = copy.deepcopy(model_at_start)
            for wt_init, grad_1_step in zip(new_params_plus_delta, model_1_step_grads):
                assert grad_1_step is not None
                wt_init.data += self.delta_HF * grad_1_step.clone()
                
            self.update_parameters(new_params_plus_delta)

            X_pp, y_pp = self.get_next_train_batch()
            self.optimizer.zero_grad()
            output = self.model(X_pp)
            loss = self.loss(output, y_pp)
            #loss.backward()
            model_plus_delta_grads = torch.autograd.grad(loss, self.model.parameters())
            
            # model_plus_delta = copy.deepcopy(list(self.model.parameters()))
            ## in model_plus_delta, we only care about grad
            
            new_params_minus_delta = copy.deepcopy(model_at_start)
            for wt_init, grad_1_step in zip(new_params_minus_delta, model_1_step_grads):
                assert grad_1_step is not None
                wt_init.data -= self.delta_HF * grad_1_step.clone()
                
            self.update_parameters(new_params_minus_delta)

            self.optimizer.zero_grad()
            output = self.model(X_pp)
            loss = self.loss(output, y_pp)
            #loss.backward()
            
            model_minus_delta_grads = torch.autograd.grad(loss, self.model.parameters())
            
            #model_minus_delta = copy.deepcopy(list(self.model.parameters()))
            ## in model_minus_delta, we only care about grad
            
            final_update = copy.deepcopy(model_at_start)
            for wt_init, grad_1_step, grad_plus_delta, grad_minus_delta in zip(final_update, model_1_step_grads, model_plus_delta_grads, model_minus_delta_grads):
                assert grad_plus_delta is not None and grad_minus_delta is not None
                wt_init.data += -self.beta_HF * (grad_1_step - self.learning_rate * (grad_plus_delta - grad_minus_delta)/(2 * self.delta_HF))
                
            # aggregate and do final update

            self.update_parameters(final_update)
            # clone model to user model 
            self.clone_model_paramenter(self.model.parameters(), self.local_model)
        return LOSS    
        
    def train_n_epochs(self, num_epochs):
        self.model.train()
        assert num_epochs is not None, "should pass in epochs as arg"
        for epoch in range(1, num_epochs + 1):
            self.model.train()

            for batch_idx, (X, y) in enumerate(self.trainloader):
                X, y = X.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(X)
                loss = self.loss(output, y) 
                loss.backward()
                self.optimizer.step()

