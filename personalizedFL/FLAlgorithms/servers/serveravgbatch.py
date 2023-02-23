import torch
import os

from FLAlgorithms.users.useravgbatch import UserAVGbatch
from FLAlgorithms.servers.serverbase import Server
from utils.model_utils import read_data, read_user_data
import numpy as np

# Implementation for FedAvg Server

class FedAvgBatch(Server):
    #NOTE: any changes to this class will propogate to peravgepoch, so be wary of this
    def __init__(self, dataset,datasetnumber, algorithm, model, batch_size, learning_rate, beta, lamda, num_glob_iters,
                 local_epochs, optimizer, num_users, times, decimate=1, device=torch.device("cpu"), validation_epochs=None):
        self.decimate = decimate
        super().__init__(dataset,datasetnumber, algorithm, model[0], batch_size, learning_rate, beta, lamda, num_glob_iters,
                         local_epochs, optimizer, num_users, times, device=device, validation_epochs=validation_epochs)

        # Initialize data for all  users
        data = read_data(dataset, datasetnumber=self.datasetnumber)
        total_users = len(data[0])
        for i in range(total_users):
            id, train, valid, test = read_user_data(i, data, dataset)
            user = UserAVGbatch(id, train, valid, test, model, batch_size, learning_rate,beta,lamda, local_epochs, optimizer, device=device, dataset=dataset)
            self.users.append(user)
            self.total_train_samples += user.train_samples
            
        print("Number of users / total users:",num_users, " / " ,total_users)
        print("Finished creating FedAvg server.")

    def send_grads(self):
        assert (self.users is not None and len(self.users) > 0)
        grads = []
        for param in self.model.parameters():
            if param.grad is None:
                grads.append(torch.zeros_like(param.data))
            else:
                grads.append(param.grad)
        for user in self.users:
            user.set_grads(grads)

    def train(self):
        loss = []
        for glob_iter in range(self.num_glob_iters):
            print("-------------Round number: ",glob_iter, " -------------")
            # Broadcast global model
            self.send_parameters()

            if glob_iter % self.decimate == 0:
                # Evaluate model each interation
                self.evaluate(epoch=glob_iter)

            self.selected_users = self.select_users(glob_iter,self.num_users)
            for user in self.selected_users:
                user.train(batches=self.local_epochs, personal=False)
            
            #aggregates updated local models
            self.aggregate_parameters()
        # self.save_results()
        # self.save_model()