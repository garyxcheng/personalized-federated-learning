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
from cifar100_utils import get_network, get_training_dataloader, get_test_dataloader, WarmUpLR, \
    most_recent_folder, most_recent_weights, last_epoch, best_acc_weights
from conf import settings

import pickle

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


#TODO
#from FedML.fedml_api.data_preprocessing.fed_cifar100.data_loader import load_partition_data_federated_cifar100

#see https://www.wandb.com/articles/multi-gpu-sweeps for details on hyperparam sweep:
# command to run CUDA_VISIBLE_DEVICES=0 taskset -c 0 wandb agent _____
import wandb

def train(args, model, device, train_loader, optimizer, epoch, cudadevice, warmup_scheduler):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        if epoch <= args.warm:
            warmup_scheduler.step()
        # if batch_idx % args.log_interval == 0:
        #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #         epoch, batch_idx * len(data), len(train_loader.dataset),
        #         100. * batch_idx / len(train_loader), loss.item()))
        #     if args.dry_run:
        #         break

def valid(model, device, valid_loader,epoch, cudadevice):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(valid_loader):
            # torch.save((data, target), f"UNIT_TEST_MATCHING_BATCH/valid_device={cudadevice}_epoch={epoch}_batch={batch_idx}.pt")
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= (len(valid_loader.dataset))

    # import pdb; pdb.set_trace()

    print('\nEpoch {} Validation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        epoch, test_loss, correct, len(valid_loader.dataset),
        100. * correct / len(valid_loader.dataset)))

    wandb.log({
        "Validation Accuracy" : 100. * correct / (len(valid_loader.dataset)),
        "Validation Loss" : test_loss,
        "Validation Number Correct" : correct
        },step=epoch)

def test(model, device, test_loader,epoch, cudadevice):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            # torch.save((data, target), f"UNIT_TEST_MATCHING_BATCH/test_device={cudadevice}_epoch={epoch}_batch={batch_idx}.pt")
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= (len(test_loader.dataset))

    print('\nEpoch {} Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        epoch, test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    wandb.log({
        "Test Accuracy" : 100. * correct / (len(test_loader.dataset)),
        "Test Loss" : test_loss,
        "Test Number Correct" : correct
        }, step=epoch)

def main():
    # # Training settings
    # parser = argparse.ArgumentParser(description='PyTorch EMNIST')
    # parser.add_argument('--batch-size', type=int, default=64, metavar='N',
    #                     help='input batch size for training (default: 64)')
    # parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
    #                     help='input batch size for testing (default: 1000)')
    # parser.add_argument('--epochs', type=int, default=25, metavar='N',
    #                     help='number of epochs to train (default: 14)')
    # parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
    #                     help='learning rate (default: 1.0)')
    # parser.add_argument('--gamma', type=float, default=0.3, metavar='M',
    #                     help='Learning rate step gamma (default: 0.7)')
    # parser.add_argument('--no-cuda', action='store_true', default=False,
    #                     help='disables CUDA training')
    # parser.add_argument('--dry-run', action='store_true', default=False,
    #                     help='quickly check a single pass')
    # parser.add_argument('--seed', type=int, default=1, metavar='S',
    #                     help='random seed (default: 1)')
    # parser.add_argument('--log-interval', type=int, default=10, metavar='N',
    #                     help='how many batches to wait before logging training status')
    # parser.add_argument('--save-model', action='store_true', default=True,
    #                     help='For Saving the current Model')

    # parser.add_argument('--dropout1', type=float, default=0.4, metavar='DO1',
    #                     help='first dropout rate (default: 0.4)')
    # parser.add_argument('--dropout2', type=float, default=0.4, metavar='DO2',
    #                     help='second dropout rate (default: 0.4)')
    # args = parser.parse_args()

    parser = argparse.ArgumentParser(description='PyTorch CIFAR100')
    #these arguments do not matter as batch size is created in jupyter notebook
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1024, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=200, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.0)')
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
    parser.add_argument('--warm', type=int, default=1, help='warm up training phase')
    parser.add_argument('--cudadevice', type=str, choices=["0", "1", "2", "3"], help='for tracking which device')
    parser.add_argument('--cudabenchmark', action='store_true', default=False,
                        help='quickly check a single pass')
    # parser.add_argument('--dropout1', type=float, default=0.4, metavar='DO1',
    #                     help='first dropout rate (default: 0.4)')
    # parser.add_argument('--dropout2', type=float, default=0.4, metavar='DO2',
    #                     help='second dropout rate (default: 0.4)')
    args = parser.parse_args()

    wandb.init(project='cifar100-P1-PRETRAIN', config=vars(args))

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    assert use_cuda

    torch.backends.cudnn.benchmark = args.cudabenchmark
    os.environ['PYTHONHASHSEED']=str(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    # train_kwargs = {'batch_size': args.batch_size}
    # test_kwargs = {'batch_size': args.test_batch_size}
    # if use_cuda:
    #     cuda_kwargs = {'num_workers': 1,
    #                    'pin_memory': True,
    #                    'shuffle': True}
    #     train_kwargs.update(cuda_kwargs)
    #     test_kwargs.update(cuda_kwargs)

    # #TODO read data
    # tup = load_partition_data_federated_cifar100(None, "./FedML/data/fed_cifar100/datasets")

    # train_loader = tup[3]
    # X_train, y_train = zip(*train_loader.dataset)
    # X_train, y_train = torch.stack(X_train), torch.stack(y_train)

    # test_loader = tup[4]
    # X_test, y_test = zip(*test_loader.dataset)
    # X_test, y_test = torch.stack(X_test), torch.stack(y_test)

    # mean, std = 0.9635, 0.1602
    # print("HARDCODE")
    # print(mean)
    # print(std)

    # print("verifying")
    # print(torch.mean(X_train, axis=tuple([0,2,3])))
    # print(torch.std(X_train, axis=tuple([0,2,3])))

    # X_train[:,0,:,:] = (X_train[:,0,:,:] - mean[0])/ std[0]
    # X_train[:,1,:,:] = (X_train[:,1,:,:] - mean[1])/ std[1]
    # X_train[:,2,:,:] = (X_train[:,2,:,:] - mean[2])/ std[2]

    # train_data =[(x,y) for x, y in zip(X_train, y_train)]
    # test_data = [(x,y) for x, y in zip(X_test, y_test)]

    # train_loader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=args.batch_size)
    # test_loader = torch.utils.data.DataLoader(test_data, shuffle=True, batch_size=args.batch_size)

    train_data_x, train_data_y, val_data_x, val_data_y, test_data_x, test_data_y = torch.load('cifar100_data_federated_split.pt')

    mean = torch.mean(train_data_x, axis=tuple([0,2,3]))
    print(mean)
    std = torch.std(train_data_x, axis=tuple([0,2,3]))
    print(std)


####################### TRANSFORMS ##########################

    train_transform = transforms.Compose([])
    train_transform.transforms.append(transforms.ToPILImage())
    train_transform.transforms.append(transforms.RandomCrop(32, padding = 4))
    train_transform.transforms.append(transforms.RandomHorizontalFlip())
    train_transform.transforms.append(transforms.RandomRotation(15))
    train_transform.transforms.append(transforms.ToTensor())
    train_transform.transforms.append(transforms.Normalize(mean=mean,std=std))

    test_transform = transforms.Compose([])
    test_transform.transforms.append(transforms.Normalize(mean=mean,std=std))

    val_transform = transforms.Compose([])
    val_transform.transforms.append(transforms.Normalize(mean=mean,std=std))


    train_dataset_aug = CustomTensorDataset(tensors=(train_data_x, train_data_y), transform=train_transform)
    train_loader = torch.utils.data.DataLoader(train_dataset_aug, batch_size=args.batch_size, shuffle=True, pin_memory = True)#, num_workers = 2)

    test_dataset_aug = CustomTensorDataset(tensors=(test_data_x, test_data_y), transform=test_transform)
    test_loader = torch.utils.data.DataLoader(test_dataset_aug, batch_size=args.test_batch_size, shuffle=False, pin_memory = True)#, num_workers = 2)

    val_dataset_aug = CustomTensorDataset(tensors=(val_data_x, val_data_y), transform=val_transform)
    valid_loader = torch.utils.data.DataLoader(val_dataset_aug, batch_size=args.test_batch_size, shuffle=False, pin_memory = True)#, num_workers = 2)


    #TODO change model
    model = resnet18().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=settings.MILESTONES, gamma=0.2) #learning rate decay
    iter_per_epoch = len(train_loader)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)

    wandb.watch(model, log="all")

    for epoch in range(1, args.epochs + 1):
        if epoch > args.warm:
            train_scheduler.step(epoch)
        train(args, model, device, train_loader, optimizer, epoch, args.cudadevice, warmup_scheduler)
        valid(model, device, valid_loader, epoch, args.cudadevice)
        test(model, device, test_loader, epoch, args.cudadevice)

    hyperparameters = [str(args.lr)]
    prefix = "_".join(hyperparameters)
    if args.save_model:
        try:
            torch.save(model.state_dict(), "cifar100_P1_pretrained_weights/"+prefix+"_cifar100_cnn_final.pt")
            wandb.save("cifar100_P1_pretrained_weights/"+prefix+'_cifar100_cnn_final.pt')
        except Exception:
            import pdb; pdb.set_trace()


if __name__ == '__main__':
    main()