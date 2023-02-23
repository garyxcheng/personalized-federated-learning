import torch
import os

from FLAlgorithms.users.usernaive import UserNaive
from FLAlgorithms.servers.serverbase import Server
from utils.model_utils import read_data, read_user_data
import numpy as np
import wandb

from tqdm import tqdm

# Implementation adapted from FedAvg Server

class ServerNaive(Server):
    def __init__(self, dataset,datasetnumber,algorithm, model, batch_size, learning_rate, beta, lamda, num_glob_iters,
                 local_epochs, optimizer, num_users, times, device=torch.device("cpu"), validation_epochs=None):
        super().__init__(dataset,datasetnumber,algorithm, model[0], batch_size, learning_rate, beta, lamda, num_glob_iters,
                         local_epochs, optimizer, num_users, times, device=device, validation_epochs=validation_epochs)
        # Initialize data for all  users
        data = read_data(dataset, datasetnumber=self.datasetnumber)
        total_users = len(data[0])
        self.total_users = total_users
        self.total_valid_samples = 0 
        self.total_test_samples = 0 
        for i in range(total_users):
            id, train, valid, test = read_user_data(i, data, dataset)
            user = UserNaive(id, train, valid, test, model, batch_size, learning_rate,beta,lamda, local_epochs, optimizer, device=device, dataset=dataset)
            self.users.append(user)
            self.total_train_samples += user.train_samples
            self.total_valid_samples += user.valid_samples
            self.total_test_samples += user.test_samples
            
        print("Number of users / total users:",num_users, " / " ,total_users)
        print("Finished creating Naive server.")

    def send_grads(self):
        # may not be used, but too scared to delete
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
        sum_valid_correct_arr = np.zeros(self.num_glob_iters)
        sum_valid_acc_arr = np.zeros(self.num_glob_iters)

        sum_test_correct_arr = np.zeros(self.num_glob_iters)
        sum_test_acc_arr = np.zeros(self.num_glob_iters)
        # run = wandb.init(project="pytorch-femnist-experiments", reinit=False, config = self.config_val)

        total_valid_samples_adjusted = 0
        total_test_samples_adjusted = 0
        for user in tqdm(self.users):
            # np.array(valid_correct_lst), total_valid_samples, np.array(test_correct_lst), total_test_samples
            valid_correct_arr, user_valid_samples_adjusted, test_correct_arr, user_test_samples_adjusted = user.train(self.num_glob_iters)
            sum_valid_correct_arr += valid_correct_arr
            sum_test_correct_arr += test_correct_arr

            total_valid_samples_adjusted += user_valid_samples_adjusted
            total_test_samples_adjusted += user_test_samples_adjusted

            valid_acc_arr = valid_correct_arr / user_valid_samples_adjusted
            test_acc_arr = test_correct_arr / user_test_samples_adjusted

            sum_valid_acc_arr += valid_acc_arr
            sum_test_acc_arr += test_acc_arr

        avg_user_valid_acc_arr = sum_valid_acc_arr / self.total_users
        avg_valid_acc_arr = sum_valid_correct_arr / total_valid_samples_adjusted

        avg_user_test_acc_arr = sum_test_acc_arr / self.total_users
        avg_test_acc_arr = sum_test_correct_arr / total_test_samples_adjusted

        assert len(avg_user_test_acc_arr) == len(avg_test_acc_arr) and len(avg_user_test_acc_arr) == len(sum_test_correct_arr)
        assert len(avg_user_valid_acc_arr) == len(avg_valid_acc_arr) and len(avg_user_test_acc_arr) == len(avg_user_valid_acc_arr)
        for idx, val in enumerate(avg_user_test_acc_arr):
            wandb.log({
                "Average User Valid Accuracy" : avg_user_valid_acc_arr[idx],
                "Average User Test Accuracy" : avg_user_test_acc_arr[idx],
                "Average Weighted Valid Accuracy" : avg_valid_acc_arr[idx],
                "Average Weighted Test Accuracy" : avg_test_acc_arr[idx],
                "Total Valid Correct" : sum_valid_correct_arr[idx],
                "Total Test Correct" : sum_test_correct_arr[idx]
            }, step=idx)