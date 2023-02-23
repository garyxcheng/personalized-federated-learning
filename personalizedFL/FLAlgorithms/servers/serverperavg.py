import torch
import numpy as np
import random 
import os

from FLAlgorithms.users.userperavg import UserPerAvg
from FLAlgorithms.servers.serverbase import Server
from utils.model_utils import read_data, read_user_data


# Implementation for per-FedAvg Server

class PerAvg(Server):
    def __init__(self, dataset,datasetnumber,algorithm, model, batch_size, learning_rate, beta, lamda, num_glob_iters,
                 local_epochs, optimizer, num_users, times, decimate=1, personal_epochs=1, device=torch.device("cpu"), validation_epochs=None):
        self.personal_epochs = personal_epochs
        self.decimate = decimate
        super().__init__(dataset,datasetnumber,algorithm, model[0], batch_size, learning_rate, beta, lamda, num_glob_iters,
                         local_epochs, optimizer, num_users, times, device=device, validation_epochs=validation_epochs)

        # Initialize data for all  users
        data = read_data(dataset, datasetnumber=self.datasetnumber)
        total_users = len(data[0])
        for i in range(total_users):
            id, train, valid, test = read_user_data(i, data, dataset)
            user = UserPerAvg(id, train, valid, test, model, batch_size, learning_rate, beta, lamda, local_epochs, optimizer, total_users , num_users, device=device, dataset=dataset)
            self.users.append(user)
            self.total_train_samples += user.train_samples
        print("Number of users / total users:",num_users, " / " ,total_users)
        print("Finished creating Local Per-Avg.")

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
            # send all parameter for users 
            self.send_parameters()

            # Evaluate gloal model on user for each interation
            if glob_iter % self.decimate == 0:
                print("Evaluate global model with one step update")
                print("")
                pytorch_state = torch.get_rng_state()
                np_state = np.random.get_state()
                random_state = random.getstate()  # remeber this state 
                
                self.evaluate_global_peravg_peravgHF_perridge(epoch=glob_iter)

                torch.set_rng_state(pytorch_state)
                np.random.set_state(np_state)
                random.setstate(random_state) 

                self.evaluate_n_epochs(self.personal_epochs, epoch=glob_iter)

            # choose several users to send back upated model to server
            self.selected_users = self.select_users(glob_iter,self.num_users)
            for user in self.selected_users:
                user.train(batches=self.local_epochs)
                
            self.aggregate_parameters()

        # self.save_results()
        # self.save_model()
