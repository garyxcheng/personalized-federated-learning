import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
from torch.utils.data import DataLoader
from FLAlgorithms.optimizers.fedoptimizer import pFedMeOptimizer
from FLAlgorithms.users.userbase import User
import copy

# Implementation for pFeMe clients
# NOTE: beta is not used here to our knowledge; the nomenclature of learning_rate and beta (like for peravg) does not hold here
class UserpFedMe(User):
    def __init__(self, numeric_id, train_data, valid_data, test_data, model, batch_size, learning_rate,beta,lamda,
                 local_epochs, optimizer, K, personal_learning_rate, device=torch.device("cpu"), dataset=None):
        assert dataset is not None
        super().__init__(numeric_id, train_data, valid_data, test_data, model[0], batch_size, learning_rate, beta, lamda,
                         local_epochs, device=device, dataset=dataset)

        self.persionalized_model = copy.deepcopy(list(self.model.parameters())) #TODO: I'm pretty sure this doesn't get used anywhere
        self.persionalized_model_bar = copy.deepcopy(list(self.model.parameters()))

        if(model[1] == "Mclr_CrossEntropy"):
            self.loss = nn.CrossEntropyLoss()
        else:
            self.loss = nn.NLLLoss()

        self.K = K
        self.personal_learning_rate = personal_learning_rate
        self.optimizer = pFedMeOptimizer(self.model.parameters(), lr=self.personal_learning_rate, lamda=self.lamda)

    def set_grads(self, new_grads):
        if isinstance(new_grads, nn.Parameter):
            for model_grad, new_grad in zip(self.model.parameters(), new_grads):
                model_grad.data = new_grad.data
        elif isinstance(new_grads, list):
            for idx, model_grad in enumerate(self.model.parameters()):
                model_grad.data = new_grads[idx]

    def train(self, epochs=None):
        self.model = self.model.to(self.device)
        try:
            self.local_model = [localweight.to(self.device) for localweight in self.local_model]
            self.persionalized_model_bar = [localweight.to(self.device) for localweight in self.persionalized_model_bar]
        except:
            import pdb; pdb.set_trace()
        # self.local_model = self.local_model.to(self.device)
        LOSS = 0
        self.model.train()
        if epochs is None:
            assert False, "should pass in epochs as arg"
            epochs = self.local_model
        for epoch in range(1, epochs + 1):  # local update
            
            self.model.train()
            X, y = self.get_next_train_batch()

            # K = 30 # K is number of personalized steps
            for i in range(self.K):
                self.optimizer.zero_grad()
                output = self.model(X)
                loss = self.loss(output, y)
                loss.backward()
                #TODO check to make sure this isn't taking too much memory; maybe need to move perionalized_model_bar to cpu
                self.persionalized_model_bar, _ = self.optimizer.step(self.local_model)

            # update local weight after finding aproximate theta
            for new_param, localweight in zip(self.persionalized_model_bar, self.local_model):
                localweight.data = localweight.data - self.lamda* self.learning_rate * (localweight.data - new_param.data)

        #update local model as local_weight_upated
        #self.clone_model_paramenter(self.local_weight_updated, self.local_model)
        self.update_parameters(self.local_model)
        self.local_model = [localweight.to(self.device) for localweight in self.local_model]
        self.persionalized_model_bar = [localweight.to(self.device) for localweight in self.persionalized_model_bar]
        # self.local_model = self.local_model.to(torch.device("cpu"))
        
        return LOSS