import torch
import numpy as np
import random 
import os

from FLAlgorithms.servers.serverbase import Server
from FLAlgorithms.servers.serveravgbatch import FedAvgBatch
from utils.model_utils import read_data, read_user_data
import numpy as np
import wandb

# Implementation adapted from serveravg
"""
We note that this personalization algorithm uses most of functionality of the FedAvg server, including the class UserAVG(User) class
whereas the other meta-learning learning algorithms are a code-clone of the PerAvg class.
In spite of the differences, the functionality is the same.
We note that PerRidgeEpoch with user_ridge_penalty = 0 behaves the same as PerAvgEpoch
"""

class PerAvgEpochBatch(FedAvgBatch):
    def __init__(self, dataset,datasetnumber,algorithm, model, batch_size, learning_rate, beta, lamda, num_glob_iters,
                 local_epochs, optimizer, num_users, times, decimate=1, personal_epochs=1, device=torch.device("cpu"), validation_epochs=None):
        self.personal_epochs = personal_epochs
        self.decimate = decimate
        super().__init__(dataset,datasetnumber,algorithm, model, batch_size, learning_rate, beta, lamda, num_glob_iters,
                 local_epochs, optimizer, num_users, times, decimate=decimate, device=device, validation_epochs=validation_epochs)
    
    def test(self):
        '''tests self.latest_model on given clients
        '''
        num_samples = []
        tot_correct = []
        losses = []
        for c in self.users:
            c.train_epochs(epochs=self.personal_epochs, personal=True)
            ct, ns = c.test()
            tot_correct.append(ct*1.0)
            num_samples.append(ns)
        ids = [c.id for c in self.users]

        return ids, num_samples, tot_correct

    def test_global_peravgepochbatch(self):
        '''tests self.latest_model on given clients
        this is just the standard test; we added this and modified test in order to preserve evaluate and wandb log consistency
        '''
        num_samples = []
        tot_correct = []
        losses = []
        for c in self.users:
            ct, ns = c.test()
            tot_correct.append(ct*1.0)
            num_samples.append(ns)
        ids = [c.id for c in self.users]

        return ids, num_samples, tot_correct

    def train(self):
        loss = []
        for glob_iter in range(self.num_glob_iters):
            print("-------------Round number: ",glob_iter, " -------------")
            if glob_iter % self.decimate == 0:
                self.send_parameters()

                # Evaluate model each iteration
                pytorch_state = torch.get_rng_state()
                np_state = np.random.get_state()
                random_state = random.getstate()  

                self.evaluate_global_peravgepochbatch(epoch=glob_iter)

                torch.set_rng_state(pytorch_state)
                np.random.set_state(np_state)
                random.setstate(random_state) 

                self.evaluate(epoch=glob_iter)

            self.send_parameters()

            self.selected_users = self.select_users(glob_iter,self.num_users)
            for user in self.selected_users:
                user.train(batches=self.local_epochs, personal=False)
            self.aggregate_parameters()
            
        # self.save_results()
        # self.save_model()

    
    def evaluate_global_peravgepochbatch(self, epoch=None):
        stats = self.test_global_peravgepochbatch() #test has to come first for peravgepoch and peravgepochbatch
        stats_valid = self.valid() 
        stats_train = self.train_error_and_loss()
        glob_acc = np.sum(stats[2])*1.0/np.sum(stats[1])
        valid_acc = np.sum(stats_valid[2])*1.0/np.sum(stats_valid[1])
        train_acc = np.sum(stats_train[2])*1.0/np.sum(stats_train[1])
        # train_loss = np.dot(stats_train[3], stats_train[1])*1.0/np.sum(stats_train[1])
        train_loss = sum([x * y for (x, y) in zip(stats_train[1], stats_train[3])]).item() / np.sum(stats_train[1])
        self.rs_glob_acc.append(glob_acc)
        self.rs_valid_acc.append(valid_acc)
        self.rs_train_acc.append(train_acc)
        self.rs_train_loss.append(train_loss)
        #print("stats_train[1]",stats_train[3][0])
        

        # Not used anymore, tried to do validation as in Adaptive federated optimization
        # Didn't actually do it because doesn't make a lot of sense for personalization.
        length_of_list = self.calculation_length_of_validation_epoch_list()
        if len(self.validation_accuracy_lst) < length_of_list:
            self.validation_accuracy_lst.append(train_loss)
        elif len(self.validation_accuracy_lst) == length_of_list:
            self.validation_accuracy_lst = self.validation_accuracy_lst[1:]
            self.validation_accuracy_lst.append(train_loss)
        else:
            raise Exception
        
        if epoch is None:
            wandb.log({
            "Global Test Accuracy" : glob_acc,
            "Global Valid Accuracy" : valid_acc,
            "Global Train Accuracy" : train_acc,
            "Global Train Loss" : train_loss,
            "Global Average of train_loss tail": np.mean(self.validation_accuracy_lst),
            "Global length of validation_accuracy_lst" : len(self.validation_accuracy_lst)
            })
        else:
            wandb.log({
            "Global Test Accuracy" : glob_acc,
            "Global Valid Accuracy" : valid_acc,
            "Global Train Accuracy" : train_acc,
            "Global Train Loss" : train_loss,
            "Global Average of train_loss tail": np.mean(self.validation_accuracy_lst),
            "Global length of validation_accuracy_lst" : len(self.validation_accuracy_lst)
            }, step=epoch)

        print("Global Average Global Accurancy: ", glob_acc)
        print("Global Average Global Valid Accurancy: ", valid_acc)
        print("Global Average Global Trainning Accurancy: ", train_acc)
        print("Global Average Global Trainning Loss: ",train_loss)
        print("Global Average of train_loss tail: ", np.mean(self.validation_accuracy_lst))
        print("Global Length of validation_accuracy_lst: ", len(self.validation_accuracy_lst))