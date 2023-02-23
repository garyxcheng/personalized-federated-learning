import torch
import os
import numpy as np
import h5py
from utils.model_utils import Metrics
import copy

import wandb

class Server:
    def __init__(self, dataset,datasetnumber,algorithm, model, batch_size, learning_rate ,beta, lamda,
                 num_glob_iters, local_epochs, optimizer,num_users, times, device=torch.device("cpu"), validation_epochs=None):
        assert validation_epochs is not None
        self.validation_epochs = validation_epochs
        self.validation_accuracy_lst = []

        # Set up the main attributes
        self.dataset = dataset
        self.datasetnumber=datasetnumber
        self.num_glob_iters = num_glob_iters
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.total_train_samples = 0
        self.model = copy.deepcopy(model)
        self.users = []
        self.selected_users = []
        self.num_users = num_users
        self.beta = beta
        self.lamda = lamda
        self.algorithm = algorithm
        self.rs_train_acc, self.rs_train_loss, self.rs_glob_acc,self.rs_train_acc_per, self.rs_train_loss_per, self.rs_glob_acc_per = [], [], [], [], [], []
        self.rs_valid_acc, self.rs_valid_acc_per = [], []
        self.times = times
        self.device = device

        # Initialize the server's grads to zeros
        #for param in self.model.parameters():
        #    param.data = torch.zeros_like(param.data)
        #    param.grad = torch.zeros_like(param.data)
        #self.send_parameters()
        
    def aggregate_grads(self):
        assert (self.users is not None and len(self.users) > 0)
        for param in self.model.parameters():
            param.grad = torch.zeros_like(param.data)
        for user in self.users:
            self.add_grad(user, user.train_samples / self.total_train_samples)

    def add_grad(self, user, ratio):
        user_grad = user.get_grads()
        for idx, param in enumerate(self.model.parameters()):
            param.grad = param.grad + user_grad[idx].clone() * ratio

    def send_parameters(self):
        assert (self.users is not None and len(self.users) > 0)
        for user in self.users:
            user.set_parameters(self.model)

    def add_parameters(self, user, ratio):
        model = self.model.parameters()
        for server_param, user_param in zip(self.model.parameters(), user.get_parameters()):
            server_param.data = server_param.data + user_param.data.clone() * ratio

    def aggregate_parameters(self):
        assert (self.users is not None and len(self.users) > 0)
        for param in self.model.parameters():
            param.data = torch.zeros_like(param.data)
        total_train = 0
        for user in self.selected_users:
            total_train += user.train_samples
        for user in self.selected_users:
            self.add_parameters(user, user.train_samples / total_train)

    def save_model(self):
        model_path = os.path.join("models", self.dataset)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(self.model, os.path.join(model_path, "server" + ".pt"))

    def load_model(self):
        model_path = os.path.join("models", self.dataset, "server" + ".pt")
        assert (os.path.exists(model_path))
        self.model = torch.load(model_path)

    def model_exists(self):
        return os.path.exists(os.path.join("models", self.dataset, "server" + ".pt"))
    
    def select_users(self, round, num_users):
        '''selects num_clients clients weighted by number of samples from possible_clients
        Args:
            num_clients: number of clients to select; default 20
                note that within function, num_clients is set to
                min(num_clients, len(possible_clients))
        
        Return:
            list of selected clients objects
        '''
        if(num_users == len(self.users)):
            print("All users are selected")
            return self.users

        num_users = min(num_users, len(self.users))
        return np.random.choice(self.users, num_users, replace=False)

    def persionalized_aggregate_parameters(self):
        # used in pFedMe
        assert (self.users is not None and len(self.users) > 0)

        # store previous parameters
        previous_param = copy.deepcopy(list(self.model.parameters()))
        for param in self.model.parameters():
            param.data = torch.zeros_like(param.data)
        total_train = 0
        #if(self.num_users = self.to)
        for user in self.selected_users:
            total_train += user.train_samples

        for user in self.selected_users:
            self.add_parameters(user, user.train_samples / total_train)
            #self.add_parameters(user, 1 / len(self.selected_users))

        # aaggregate avergage model with previous model using parameter beta 
        for pre_param, param in zip(previous_param, self.model.parameters()):
            param.data = (1 - self.beta)*pre_param.data + self.beta*param.data
            
    # Save loss, accurancy to h5 fiel
    def save_results(self):
        alg = self.dataset + "_" + self.algorithm
        alg = alg + "_" + str(self.learning_rate) + "_" + str(self.beta) + "_" + str(self.lamda) + "_" + str(self.num_users) + "u" + "_" + str(self.batch_size) + "b" + "_" + str(self.local_epochs)
        if(self.algorithm == "pFedMe" or self.algorithm == "pFedMe_p"):
            alg = alg + "_" + str(self.K) + "_" + str(self.personal_learning_rate)
        alg = alg + "_" + str(self.times)
        if (len(self.rs_glob_acc) != 0 &  len(self.rs_train_acc) & len(self.rs_train_loss)) :
            with h5py.File("./results/"+'{}.h5'.format(alg, self.local_epochs), 'w') as hf:
                hf.create_dataset('rs_glob_acc', data=self.rs_glob_acc)
                hf.create_dataset('rs_train_acc', data=self.rs_train_acc)
                hf.create_dataset('rs_train_loss', data=self.rs_train_loss)
                hf.close()
        
        # store persionalized value
        alg = self.dataset + "_" + self.algorithm + "_p"
        alg = alg  + "_" + str(self.learning_rate) + "_" + str(self.beta) + "_" + str(self.lamda) + "_" + str(self.num_users) + "u" + "_" + str(self.batch_size) + "b"+ "_" + str(self.local_epochs)
        if(self.algorithm == "pFedMe" or self.algorithm == "pFedMe_p"):
            alg = alg + "_" + str(self.K) + "_" + str(self.personal_learning_rate)
        alg = alg + "_" + str(self.times)
        if (len(self.rs_glob_acc_per) != 0 &  len(self.rs_train_acc_per) & len(self.rs_train_loss_per)) :
            with h5py.File("./results/"+'{}.h5'.format(alg, self.local_epochs), 'w') as hf:
                hf.create_dataset('rs_glob_acc', data=self.rs_glob_acc_per)
                hf.create_dataset('rs_train_acc', data=self.rs_train_acc_per)
                hf.create_dataset('rs_train_loss', data=self.rs_train_loss_per)
                hf.close()

    def test(self):
        '''tests self.latest_model on given clients
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
    
    def valid(self):
        '''tests self.latest_model on given clients
        '''
        num_samples = []
        tot_correct = []
        losses = []
        for c in self.users:
            ct, ns = c.valid()
            tot_correct.append(ct*1.0)
            num_samples.append(ns)
        ids = [c.id for c in self.users]

        return ids, num_samples, tot_correct

    def train_error_and_loss(self):
        num_samples = []
        tot_correct = []
        losses = []
        for c in self.users:
            ct, cl, ns = c.train_error_and_loss() 
            tot_correct.append(ct*1.0)
            num_samples.append(ns)
            losses.append(cl*1.0)
        
        ids = [c.id for c in self.users]
        #groups = [c.group for c in self.clients]

        return ids, num_samples, tot_correct, losses

    def test_persionalized_model(self):
        # used only in pFedMe
        '''tests self.latest_model on given clients
        '''
        num_samples = []
        tot_correct = []
        for c in self.users:
            ct, ns = c.test_persionalized_model()
            tot_correct.append(ct*1.0)
            num_samples.append(ns)
        ids = [c.id for c in self.users]

        return ids, num_samples, tot_correct

    def valid_persionalized_model(self):
        # used only in pFedMe
        '''tests self.latest_model on given clients
        '''
        num_samples = []
        tot_correct = []
        for c in self.users:
            ct, ns = c.valid_persionalized_model()
            tot_correct.append(ct*1.0)
            num_samples.append(ns)
        ids = [c.id for c in self.users]

        return ids, num_samples, tot_correct

    def train_error_and_loss_persionalized_model(self):
        # used only in pFedMe
        num_samples = []
        tot_correct = []
        losses = []
        for c in self.users:
            ct, cl, ns = c.train_error_and_loss_persionalized_model() 
            tot_correct.append(ct*1.0)
            num_samples.append(ns)
            losses.append(cl*1.0)
        
        ids = [c.id for c in self.users]
        #groups = [c.group for c in self.clients]

        return ids, num_samples, tot_correct, losses

    def calculation_length_of_validation_epoch_list(self):
        length_of_list = -1
        try:
            if self.validation_epochs % self.decimate == 0:
                length_of_list = int(self.validation_epochs / self.decimate + 1)
            else:
                length_of_list = int(np.ceil(self.validation_epochs / self.decimate))
        except AttributeError:
            length_of_list = self.validation_epochs
        assert length_of_list > 0 
        assert isinstance(length_of_list, int)
        return length_of_list

    def evaluate(self, epoch=None):
        stats = self.test() #test has to come first for peravgepoch and peravgepochbatch
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
            "Test Accuracy" : glob_acc,
            "Valid Accuracy" : valid_acc,
            "Train Accuracy" : train_acc,
            "Train Loss" : train_loss,
            "Average of train_loss tail": np.mean(self.validation_accuracy_lst),
            "length of validation_accuracy_lst" : len(self.validation_accuracy_lst)
            })
        else:
            wandb.log({
            "Test Accuracy" : glob_acc,
            "Valid Accuracy" : valid_acc,
            "Train Accuracy" : train_acc,
            "Train Loss" : train_loss,
            "Average of train_loss tail": np.mean(self.validation_accuracy_lst),
            "length of validation_accuracy_lst" : len(self.validation_accuracy_lst)
            }, step=epoch)

        print("Average Global Accurancy: ", glob_acc)
        print("Average Global Valid Accurancy: ", valid_acc)
        print("Average Global Trainning Accurancy: ", train_acc)
        print("Average Global Trainning Loss: ",train_loss)
        print("Average of train_loss tail: ", np.mean(self.validation_accuracy_lst))
        print("Length of validation_accuracy_lst: ", len(self.validation_accuracy_lst))
    

    def evaluate_global_peravg_peravgHF_perridge(self, epoch=None):
        #used to compute central model accuracy in  peravg_peravgHF_perridge

        stats = self.test() #test has to come first for peravgepoch and peravgepochbatch
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

    def evaluate_personalized_model(self, epoch=None):
        # used only in pFedMe 
        stats = self.test_persionalized_model()  
        stats_valid = self.valid_persionalized_model() 
        stats_train = self.train_error_and_loss_persionalized_model()
        glob_acc = np.sum(stats[2])*1.0/np.sum(stats[1])
        valid_acc = np.sum(stats_valid[2])*1.0/np.sum(stats_valid[1])
        train_acc = np.sum(stats_train[2])*1.0/np.sum(stats_train[1])
        # train_loss = np.dot(stats_train[3], stats_train[1])*1.0/np.sum(stats_train[1])
        train_loss = sum([x * y for (x, y) in zip(stats_train[1], stats_train[3])]).item() / np.sum(stats_train[1])
        self.rs_glob_acc_per.append(glob_acc)
        self.rs_valid_acc_per.append(valid_acc)
        self.rs_train_acc_per.append(train_acc)
        self.rs_train_loss_per.append(train_loss)
        #print("stats_train[1]",stats_train[3][0])

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
            "pFedMe Test Accuracy" : glob_acc,
            "pFedMe Valid Accuracy" : valid_acc,
            "pFedMe Train Accuracy" : train_acc,
            "pFedMe Train Loss" : train_loss,
            "pFedMe Average of train_loss tail": np.mean(self.validation_accuracy_lst),
            "pFedMe length of validation_accuracy_lst" : len(self.validation_accuracy_lst)
            })
        else:
            wandb.log({
            "pFedMe Test Accuracy" : glob_acc,
            "pFedMe Valid Accuracy" : valid_acc,
            "pFedMe Train Accuracy" : train_acc,
            "pFedMe Train Loss" : train_loss,
            "pFedMe Average of train_loss tail": np.mean(self.validation_accuracy_lst),
            "pFedMe length of validation_accuracy_lst" : len(self.validation_accuracy_lst)
            }, step=epoch)
        print("Average Personal Accurancy: ", glob_acc)
        print("Average Personal Training Accurancy: ", train_acc)
        print("Average Personal Training Loss: ", train_loss)
        print("Average of train_loss tail: ", np.mean(self.validation_accuracy_lst))
        print("Length of validation_accuracy_lst: ", len(self.validation_accuracy_lst))

    def evaluate_n_epochs(self, num_epoch, epoch=None):
        for c in self.users:
            # c.train_one_step() OLD
            c.train_n_epochs(num_epochs=num_epoch)

        stats = self.test()  
        stats_valid = self.valid() 
        stats_train = self.train_error_and_loss()

        # set local model back to client for training process.
        for c in self.users:
            c.update_parameters(c.local_model)

        glob_acc = np.sum(stats[2])*1.0/np.sum(stats[1])
        valid_acc = np.sum(stats_valid[2])*1.0/np.sum(stats_valid[1])
        train_acc = np.sum(stats_train[2])*1.0/np.sum(stats_train[1])
        # train_loss = np.dot(stats_train[3], stats_train[1])*1.0/np.sum(stats_train[1])
        train_loss = sum([x * y for (x, y) in zip(stats_train[1], stats_train[3])]).item() / np.sum(stats_train[1])
        self.rs_glob_acc_per.append(glob_acc)
        self.rs_valid_acc.append(valid_acc)
        self.rs_train_acc_per.append(train_acc)
        self.rs_train_loss_per.append(train_loss)
        #print("stats_train[1]",stats_train[3][0])

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
            "Test Accuracy" : glob_acc,
            "Valid Accuracy" : valid_acc,
            "Train Accuracy" : train_acc,
            "Train Loss" : train_loss,
            "Average of train_loss tail": np.mean(self.validation_accuracy_lst),
            "length of validation_accuracy_lst" : len(self.validation_accuracy_lst)
            })
        else: 
            wandb.log({
            "Test Accuracy" : glob_acc,
            "Valid Accuracy" : valid_acc,
            "Train Accuracy" : train_acc,
            "Train Loss" : train_loss,
            "Average of train_loss tail": np.mean(self.validation_accuracy_lst),
            "length of validation_accuracy_lst" : len(self.validation_accuracy_lst)
            }, step=epoch)

        print("Average Personal Accurancy: ", glob_acc)
        print("Average Personal Valid Accurancy: ", valid_acc)
        print("Average Personal Trainning Accurancy: ", train_acc)
        print("Average Personal Trainning Loss: ",train_loss)
        print("Average of train_loss tail: ", np.mean(self.validation_accuracy_lst))
        print("Length of validation_accuracy_lst: ", len(self.validation_accuracy_lst))

    def evaluate_n_batches(self, num_batches, epoch=None):
        for c in self.users:
            # c.train_one_step() OLD
            c.train_n_batches(num_batches=num_batches)

        stats = self.test()  
        stats_valid = self.valid() 
        stats_train = self.train_error_and_loss()

        # set local model back to client for training process.
        for c in self.users:
            c.update_parameters(c.local_model)

        glob_acc = np.sum(stats[2])*1.0/np.sum(stats[1])
        valid_acc = np.sum(stats_valid[2])*1.0/np.sum(stats_valid[1])
        train_acc = np.sum(stats_train[2])*1.0/np.sum(stats_train[1])
        # train_loss = np.dot(stats_train[3], stats_train[1])*1.0/np.sum(stats_train[1])
        train_loss = sum([x * y for (x, y) in zip(stats_train[1], stats_train[3])]).item() / np.sum(stats_train[1])
        self.rs_glob_acc_per.append(glob_acc)
        self.rs_valid_acc.append(valid_acc)
        self.rs_train_acc_per.append(train_acc)
        self.rs_train_loss_per.append(train_loss)
        #print("stats_train[1]",stats_train[3][0])

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
            "Test Accuracy" : glob_acc,
            "Valid Accuracy" : valid_acc,
            "Train Accuracy" : train_acc,
            "Train Loss" : train_loss,
            "Average of train_loss tail": np.mean(self.validation_accuracy_lst),
            "length of validation_accuracy_lst" : len(self.validation_accuracy_lst)
            })
        else: 
            wandb.log({
            "Test Accuracy" : glob_acc,
            "Valid Accuracy" : valid_acc,
            "Train Accuracy" : train_acc,
            "Train Loss" : train_loss,
            "Average of train_loss tail": np.mean(self.validation_accuracy_lst),
            "length of validation_accuracy_lst" : len(self.validation_accuracy_lst)
            }, step=epoch)

        print("Average Personal Accurancy: ", glob_acc)
        print("Average Personal Valid Accurancy: ", valid_acc)
        print("Average Personal Trainning Accurancy: ", train_acc)
        print("Average Personal Trainning Loss: ",train_loss)
        print("Average of train_loss tail: ", np.mean(self.validation_accuracy_lst))
        print("Length of validation_accuracy_lst: ", len(self.validation_accuracy_lst))