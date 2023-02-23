import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
from torch.utils.data import DataLoader
from FLAlgorithms.users.userbase import User

# Implementation for FedAvg clients
# beta is global level learning rate; learning_rate is personal level learning rate

class UserAVGbatch(User):
    def __init__(self, numeric_id, train_data, valid_data, test_data, model, batch_size, learning_rate, beta, lamda,
                 local_epochs, optimizer, device=torch.device("cpu"), dataset=None):
        assert dataset is not None
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

    def train(self, batches=None, personal=False):
        #if personal is True, then use the personal optimizer with beta learning rate
        # COMMENTED OUT BECAUSE PERAVGEPOCHBATCHONESTEP NEEDS PERSONAL=TRUE OPTION ; ALL P2,P3,P4 SWEEPS HAD THIS ASSERT STATEMENT HERE assert not personal, "in this version; personalization steps call train_epochs"
        if personal:
            self.personal_optimizer, self.optimizer = self.optimizer, self.personal_optimizer

        LOSS = 0
        self.model.train()
        if batches is None:
            assert False, "should pass in epochs as arg"
        for batch in range(1, batches + 1):
            self.model.train()

            # the line below was added here by me; the way batching is done in the original code is different; 
            # however, federated averaging the way it was proposed is to do actual epochs as we have it now
            # for batch_idx, (X, y) in enumerate(self.trainloader):
            X, y = self.get_next_train_batch()
            X, y = X.to(self.device), y.to(self.device) #this is technically not necessary
            self.optimizer.zero_grad()
            output = self.model(X)
            loss = self.loss(output, y) 
            loss.backward()
            self.optimizer.step()
        self.clone_model_paramenter(self.model.parameters(), self.local_model)
        # Dont know why we do this, just keeping from old code

        if personal:
            self.personal_optimizer, self.optimizer = self.optimizer, self.personal_optimizer
        return LOSS

    def train_epochs(self, epochs=None, personal=False):
        #if personal is True, then use the personal optimizer with beta learning rate
        if personal:
            self.personal_optimizer, self.optimizer = self.optimizer, self.personal_optimizer

        LOSS = 0
        self.model.train()
        if epochs is None:
            assert False, "should pass in epochs as arg"
        for epoch in range(1, epochs + 1):
            self.model.train()

            # the line below was added here by me; the way batching is done in the original code is different; 
            # however, federated averaging the way it was proposed is to do actual epochs as we have it now
            for batch_idx, (X, y) in enumerate(self.trainloader):
                X, y = X.to(self.device), y.to(self.device)
            # X, y = self.get_next_train_batch()
                self.optimizer.zero_grad()
                output = self.model(X)
                loss = self.loss(output, y) 
                loss.backward()
                self.optimizer.step()
            self.clone_model_paramenter(self.model.parameters(), self.local_model)

        if personal:
            self.personal_optimizer, self.optimizer = self.optimizer, self.personal_optimizer
        return LOSS
