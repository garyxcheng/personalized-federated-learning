#!/usr/bin/env python
import h5py
import matplotlib.pyplot as plt
import numpy as np
import argparse
import importlib
import random
import os
from FLAlgorithms.servers.serveravgbatch import FedAvgBatch
from FLAlgorithms.servers.serverpFedMe import pFedMe
from FLAlgorithms.servers.serverperavg import PerAvg
from FLAlgorithms.servers.serverperavgHF import PerAvgHF
from FLAlgorithms.servers.serverperavgHFonestep import PerAvgHFOneStep
from FLAlgorithms.servers.serverperavgepochbatch import PerAvgEpochBatch
from FLAlgorithms.servers.serverperavgepochbatchonestep import PerAvgEpochBatchOneStep
from FLAlgorithms.servers.serverperridgeepochbatch import PerRidgeEpochBatch
from FLAlgorithms.servers.servernaive import ServerNaive
from FLAlgorithms.servers.serverperavgonestep import PerAvgOneStep
from FLAlgorithms.trainmodel.models import *
from utils.plot_utils import *
import torch
torch.manual_seed(0)

import wandb

def main(dataset, datasetnumber, algorithm, model, batch_size, learning_rate, beta, lamda, num_glob_iters,
         local_epochs, optimizer, numusers, K, personal_learning_rate, times, no_cuda, seed, personal_epochs, decimate, rank, user_ridge_penalty,
         delta_HF, validation_epochs):
    assert args.seed == seed, (args.seed, seed)
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED']=str(args.seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(seed)
    random.seed(args.seed)
    assert args.seed == seed, (args.seed, seed)

    assert decimate >= 1
    if decimate > 1:
        assert num_glob_iters % decimate != 0, "should add a glob iter to track last iteration accuracy"

    assert times == 1
    for i in range(times):
        print("---------------Running time:------------",i)
        use_cuda = not no_cuda and torch.cuda.is_available()
        print(use_cuda)

        assert use_cuda, "currently data loaders created in userbase.p have pin_memory=True; might need to change to False if using cpu"
        device = torch.device("cuda" if use_cuda else "cpu")
        cpu_device = torch.device("cpu")

        # Generate model
        if model == "rnn":
            if dataset == "prep_Shakespeare":
                model = Last_Layer_Shakespeare_Net().to(device), model
            else:
                raise Exception
        elif model=="rnnstackoverflow":
            if dataset == "prep_Stackoverflownwp":
                model = Last_Layer_Stackoverflownwp_Net().to(device), model
                assert model[0].__class__.__name__ == "Last_Layer_Stackoverflownwp_Net", model[0]
            else:
                raise Exception
        elif model == "resnet":
            if dataset == "prep_CIFAR100":
                model = Last_Layer_CIFAR100_Net().to(device), model
            else:
                raise Exception     
        elif(model == "cnn"):
            if dataset == "prep_FederatedEMNIST" or dataset == "prep_Femnist":
                model = Last_Layer_EMNIST_Net().to(device), model
            else:
                raise Exception
        else:
            raise Exception


        # select algorithm
        if(algorithm == "FedAvgBatch"):
            server = FedAvgBatch(dataset, datasetnumber, algorithm, model, batch_size, learning_rate, beta, lamda, num_glob_iters, local_epochs, optimizer, numusers, i, decimate=decimate, device=device, validation_epochs=validation_epochs)
        elif(algorithm == "pFedMe"):
            server = pFedMe(dataset, datasetnumber, algorithm, model, batch_size, learning_rate, beta, lamda, num_glob_iters, local_epochs, optimizer, numusers, K, personal_learning_rate, i, decimate=decimate, device=device, validation_epochs=validation_epochs)
        elif(algorithm == "PerAvg"):
            server = PerAvg(dataset, datasetnumber, algorithm, model, batch_size, learning_rate, beta, lamda, num_glob_iters, local_epochs, optimizer, numusers, i, decimate=decimate, personal_epochs=personal_epochs, device=device, validation_epochs=validation_epochs)
        elif(algorithm == "PerAvgEpochBatch"):
            server = PerAvgEpochBatch(dataset, datasetnumber, algorithm, model, batch_size, learning_rate, beta, lamda, num_glob_iters, local_epochs, optimizer, numusers, i, decimate=decimate, personal_epochs=personal_epochs, device=device, validation_epochs=validation_epochs)
        elif(algorithm == "PerAvgEpochBatchOneStep"):
            server = PerAvgEpochBatchOneStep(dataset, datasetnumber, algorithm, model, batch_size, learning_rate, beta, lamda, num_glob_iters, local_epochs, optimizer, numusers, i, decimate=decimate, personal_epochs=personal_epochs, device=device, validation_epochs=validation_epochs)
        elif algorithm == "PerRidgeEpochBatch":
            server = PerRidgeEpochBatch(dataset, datasetnumber, algorithm, model, batch_size, learning_rate, beta, lamda, num_glob_iters, local_epochs, optimizer, numusers, i, decimate=decimate, personal_epochs=personal_epochs, device=device, user_ridge_penalty=user_ridge_penalty, validation_epochs=validation_epochs)
        elif algorithm == "PerAvgHF":
            server = PerAvgHF(dataset, datasetnumber, algorithm, model, batch_size, learning_rate, beta, lamda, num_glob_iters, local_epochs, optimizer, numusers, i, beta_HF=beta, delta_HF=delta_HF, decimate=decimate, personal_epochs=personal_epochs, device=device, validation_epochs=validation_epochs)
        elif algorithm == "PerAvgHFOneStep":
            server = PerAvgHFOneStep(dataset, datasetnumber, algorithm, model, batch_size, learning_rate, beta, lamda, num_glob_iters, local_epochs, optimizer, numusers, i, beta_HF=beta, delta_HF=delta_HF, decimate=decimate, personal_epochs=personal_epochs, device=device, validation_epochs=validation_epochs)
        elif algorithm == "Naive":
            server = ServerNaive(dataset, datasetnumber, algorithm, model, batch_size, learning_rate, beta, lamda, num_glob_iters, local_epochs, optimizer, numusers, i, device=device, validation_epochs=validation_epochs)
        elif algorithm == "PerAvgOneStep":
            server = PerAvgOneStep(dataset, datasetnumber, algorithm, model, batch_size, learning_rate, beta, lamda, num_glob_iters, local_epochs, optimizer, numusers, i, decimate=decimate, personal_epochs=personal_epochs, device=device, validation_epochs=validation_epochs)
        else:
            raise Exception
        server.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="prep_FederatedEMNIST", choices=["prep_FederatedEMNIST", "prep_Femnist", "prep_Shakespeare", "prep_CIFAR100", "prep_Stackoverflownwp"])
    parser.add_argument("--datasetnumber", type=int, default=0, help="denotes which version of the train-validation split is used")
    parser.add_argument("--model", type=str, default="cnn", choices=["cnn", "rnn", "resnet", "rnnstackoverflow"])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=0.005, help="Local learning rate")
    parser.add_argument("--beta", type=float, default=1.0, help="Average moving parameter for pFedMe, or Second learning rate of Per-FedAvg")
    parser.add_argument("--lamda", type=int, default=15, help="Regularization term")
    parser.add_argument("--num_global_iters", type=int, default=800)
    parser.add_argument("--local_epochs", type=int, default=20)
    parser.add_argument("--optimizer", type=str, default="SGD")
    parser.add_argument("--algorithm", type=str, default="pFedMe",choices=["pFedMe", "PerAvg", "FedAvg", "FedAvgBatch", "PerAvgEpochBatch",  "PerAvgHF", "Naive", "PerRidgeEpochBatch", "PerAvgOneStep", "PerAvgEpochBatchOneStep", "PerAvgHFOneStep"]) 
    parser.add_argument("--numusers", type=int, default=20, help="Number of Users per round")
    parser.add_argument("--K", type=int, default=5, help="Computation steps")
    parser.add_argument("--personal_learning_rate", type=float, default=0.09, help="Persionalized learning rate to caculate theta aproximately using K steps")
    parser.add_argument("--times", type=int, default=1, help="running time")
    parser.add_argument("--personal_epochs", type=int, default=1, help="running time")
    parser.add_argument("--decimate", type=int, default=1, help="decimate")
    parser.add_argument("--rank", type=int, default=0, help="rank")
    parser.add_argument("--user_ridge_penalty", type=float, default=0.0, help="user-ridge-penalty")
    parser.add_argument("--delta_HF", type=float, default=0.001, help="delta_HF for HF alg")
    parser.add_argument("--validation_epochs", type=int, default=100, help="number of validation epochs for hyperparameter tuning")

    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--wandbgroup', type=str, default="sweep 0",
                        help='group parameter for wandb initialization')
    args = parser.parse_args()

    if args.dataset == "prep_Shakespeare":
        wandb.init(project="shakespeare-P2-SWEEPS", config=vars(args), group=args.wandbgroup)
    elif args.dataset == "prep_Stackoverflownwp":
        wandb.init(project="stackoverflownwp-P2-sweeps", config=vars(args), group=args.wandbgroup)
    elif args.dataset == "prep_CIFAR100":
        wandb.init(project="cifar100-P2-SWEEPS", config=vars(args), group=args.wandbgroup)
    elif args.dataset == "prep_FederatedEMNIST":
        wandb.init(project="federatedemnist-P2-sweeps", config=vars(args), group=args.wandbgroup)

    print("=" * 80)
    print("Summary of training process:")
    print("Algorithm: {}".format(args.algorithm))
    print("Batch size: {}".format(args.batch_size))
    print("Learing rate       : {}".format(args.learning_rate))
    print("Average Moving       : {}".format(args.beta))
    print("Subset of users      : {}".format(args.numusers))
    print("Number of global rounds       : {}".format(args.num_global_iters))
    print("Number of local rounds       : {}".format(args.local_epochs))
    print("Dataset       : {}".format(args.dataset))
    print("Dataset Number      : {}".format(args.datasetnumber))
    print("Local Model       : {}".format(args.model))
    print("Use Cuda       : {}".format(not args.no_cuda and torch.cuda.is_available()))
    print("Seed       : {}".format(args.seed))
    print("Rank       : {}".format(args.rank))
    print("user_ridge_penalty       : {}".format(args.user_ridge_penalty))
    print("delta_HF       : {}".format(args.delta_HF))
    print("validation_epochs       : {}".format(args.validation_epochs))
    print("=" * 80)

    main(
        dataset=args.dataset,
        datasetnumber=args.datasetnumber,
        algorithm = args.algorithm,
        model=args.model,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        beta = args.beta, 
        lamda = args.lamda,
        num_glob_iters=args.num_global_iters,
        local_epochs=args.local_epochs,
        optimizer= args.optimizer,
        numusers = args.numusers,
        K=args.K,
        personal_learning_rate=args.personal_learning_rate,
        times = args.times,
        no_cuda= args.no_cuda,
        seed=args.seed,
        personal_epochs=args.personal_epochs,
        decimate=args.decimate,
        rank = args.rank,
        user_ridge_penalty = args.user_ridge_penalty,
        delta_HF = args.delta_HF,
        validation_epochs = args.validation_epochs
    )
