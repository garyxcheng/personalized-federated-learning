import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
from torch.utils.data import DataLoader
from FLAlgorithms.users.userbase import User
import numpy as np
# Implementation for Naive clients
# Adapted from useravg

class UserNaive(User):
    def __init__(self, numeric_id, train_data, valid_data, test_data, model, batch_size, learning_rate, beta, lamda,
                 local_epochs, optimizer, device=torch.device("cpu"), dataset=None):
        assert dataset is not None
        super().__init__(numeric_id, train_data, valid_data, test_data, model[0], batch_size, learning_rate, beta, lamda,
                         local_epochs, device=device, dataset=dataset)

        self.loss = nn.NLLLoss()

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.beta)

    def set_grads(self, new_grads):
        if isinstance(new_grads, nn.Parameter):
            for model_grad, new_grad in zip(self.model.parameters(), new_grads):
                model_grad.data = new_grad.data
        elif isinstance(new_grads, list):
            for idx, model_grad in enumerate(self.model.parameters()):
                model_grad.data = new_grads[idx]

    def train(self, epochs): 
        LOSS = 0
        self.model.train()
        test_correct_lst = []
        valid_correct_lst = []
        for epoch in range(epochs):
            self.model.train()

            for batch_idx, (X, y) in enumerate(self.trainloader):
                X, y = X.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(X)
                loss = self.loss(output, y)
                loss.backward()
                self.optimizer.step()
            with torch.no_grad():
                test_correct, user_test_samples_adjusted = self.test()
                valid_correct, user_valid_samples_adjusted = self.valid()
                test_correct_lst.append(test_correct)
                valid_correct_lst.append(valid_correct)
        return np.array(valid_correct_lst), user_valid_samples_adjusted, np.array(test_correct_lst), user_test_samples_adjusted



    # #TODO we use the functionality from userbase now
    # def test(self):
    #     self.model.eval()
    #     test_acc = 0
    #     for x, y in self.testloaderfull:
    #         x, y = x.to(self.device), y.to(self.device)
    #         output = self.model(x)
    #         test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
    #     return test_acc