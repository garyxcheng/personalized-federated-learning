#############
# THIS IS THE NEW VERSION ASSOCIATED WITH FEDML
#############
from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from cnn import *
import os
import json
import numpy as np
import random

#TODO
# from FedML.fedml_api.data_preprocessing.FederatedEMNIST.data_loader import load_partition_data_federated_emnist


from torch.utils.data import Dataset
#TODO do something about this
class CustomTensorDataset(Dataset):
    """TensorDataset with support of transforms.
    """
    def __init__(self, tensors, transform=None):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.transform = transform

    def __getitem__(self, index):
        x = self.tensors[0][index]

        if self.transform:
            x = self.transform(x)

        y = self.tensors[1][index]

        return x, y

    def __len__(self):
        return self.tensors[0].size(0)


#see https://www.wandb.com/articles/multi-gpu-sweeps for details on hyperparam sweep:
# command to run CUDA_VISIBLE_DEVICES=0 taskset -c 0 wandb agent _____
import wandb

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        # if batch_idx % args.log_interval == 0:
        #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #         epoch, batch_idx * len(data), len(train_loader.dataset),
        #         100. * batch_idx / len(train_loader), loss.item()))
        #     if args.dry_run:
        #         break

def valid(model, device, valid_loader, epoch):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in valid_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(valid_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(valid_loader.dataset),
        100. * correct / len(valid_loader.dataset)))

    wandb.log({
        "Validation Accuracy" : 100. * correct / len(valid_loader.dataset),
        "Validation Loss" : test_loss,
        "Validation Number Correct" : correct
        }, step=epoch)

def test(model, device, test_loader, epoch):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    wandb.log({
        "Test Accuracy" : 100. * correct / len(test_loader.dataset),
        "Test Loss" : test_loss,
        "Test Number Correct" : correct
        }, step=epoch)

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch EMNIST')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1024, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=25, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.3, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    parser.add_argument('--cudabenchmark', action='store_true', default=False,
                        help='quickly check a single pass')

    parser.add_argument('--wandbgroup', type=str, help='wandbgroupname')

    parser.add_argument('--dropout1', type=float, default=0.4, metavar='DO1',
                        help='first dropout rate (default: 0.4)')
    parser.add_argument('--dropout2', type=float, default=0.4, metavar='DO2',
                        help='second dropout rate (default: 0.4)')
    args = parser.parse_args()
    wandb.init(project="federatedemnist-P1-PRETRAIN", config=vars(args), group=args.wandbgroup)

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    assert use_cuda

    os.environ['PYTHONHASHSEED']=str(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    #### OLD VERSION
    # train_kwargs = {'batch_size': args.batch_size}
    # test_kwargs = {'batch_size': args.test_batch_size}
    # if use_cuda:
    #     cuda_kwargs = {'num_workers': 1,
    #                    'pin_memory': True,
    #                    'shuffle': True}
    #     train_kwargs.update(cuda_kwargs)
    #     test_kwargs.update(cuda_kwargs)

    #TODO read data
    # tup = load_partition_data_federated_emnist(None, "./FedML/data/FederatedEMNIST/datasets")
  
    # train_loader = tup[3]
    # X_train, y_train = zip(*train_loader.dataset)
    # X_train, y_train = torch.stack(X_train), torch.stack(y_train)

    # test_loader = tup[4]
    # X_test, y_test = zip(*test_loader.dataset)
    # X_test, y_test = torch.stack(X_test), torch.stack(y_test)

    # mean, std = 0.9635, 0.1602
    # X_train = (X_train - mean) / std
    # X_test = (X_test - mean) / std

    # train_data =[(x,y) for x, y in zip(X_train, y_train)]
    # test_data = [(x,y) for x, y in zip(X_test, y_test)]

    # train_loader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=args.batch_size)
    # test_loader = torch.utils.data.DataLoader(test_data, shuffle=True, batch_size=args.batch_size)
    #######

    train_data_x, train_data_y, val_data_x, val_data_y, test_data_x, test_data_y = torch.load('FederatedEMNIST_data_federated_split.pt')

    train_dataset_aug = CustomTensorDataset(tensors=(train_data_x, train_data_y))
    train_loader = torch.utils.data.DataLoader(train_dataset_aug, batch_size=args.batch_size, shuffle=True, pin_memory = True)#, num_workers = 2)

    test_dataset_aug = CustomTensorDataset(tensors=(test_data_x, test_data_y))
    test_loader = torch.utils.data.DataLoader(test_dataset_aug, batch_size=512, shuffle=False, pin_memory = True)#, num_workers = 2)

    val_dataset_aug = CustomTensorDataset(tensors=(val_data_x, val_data_y))
    val_loader = torch.utils.data.DataLoader(val_dataset_aug, batch_size=512, shuffle=False, pin_memory = True)#, num_workers = 2)

    model = FederatedEMNIST_Net(args.dropout1, args.dropout2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    wandb.watch(model, log="all")

    for epoch in range(args.epochs):
        train(args, model, device, train_loader, optimizer, epoch)
        valid(model, device, val_loader, epoch)
        test(model, device, test_loader, epoch)
        scheduler.step()

    hyperparameters = [str(args.lr), str(args.gamma)]
    prefix = "_".join(hyperparameters)
    if args.save_model:
        torch.save(model.state_dict(), "federatedemnist_P1_pretrain_weights/"+prefix+"_federatedemnist_cnn.pt")
        wandb.save("federatedemnist_P1_pretrain_weights/" + prefix+'_federatedemnist_cnn.pt')

if __name__ == '__main__':
    main()