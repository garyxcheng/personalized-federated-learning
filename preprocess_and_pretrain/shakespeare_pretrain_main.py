from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from rnn import *
import os
import json
import numpy as np
import random

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

# from FedML.fedml_api.data_preprocessing.fed_shakespeare.data_loader import load_partition_data_federated_shakespeare


#see https://www.wandb.com/articles/multi-gpu-sweeps for details on hyperparam sweep:
# command to run CUDA_VISIBLE_DEVICES=0 taskset -c 0 wandb agent _____
import wandb


def train(args, model, device, train_loader, optimizer, epoch, cudadevice):
    model.train()
    # import pdb; pdb.set_trace()
    for batch_idx, (data, target) in enumerate(train_loader):
        # torch.save((data, target), f"UNIT_TEST_MATCHING_BATCH/train_device={cudadevice}_epoch={epoch}_batch={batch_idx}.pt")
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        # import pdb; pdb.set_trace()

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
    test_loss /= (len(valid_loader.dataset) * 80)

    # import pdb; pdb.set_trace()

    print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(valid_loader.dataset),
        100. * correct / len(valid_loader.dataset)))

    wandb.log({
        "Validation Accuracy" : 100. * correct / (80 * len(valid_loader.dataset)),
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
    test_loss /= (len(test_loader.dataset) * 80)

    print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    wandb.log({
        "Test Accuracy" : 100. * correct / (80 * len(test_loader.dataset)),
        "Test Loss" : test_loss,
        "Test Number Correct" : correct
        }, step=epoch)

def main():
    ########################################
    ## VERY OLD REFERENCE STUFF
    ########################################
    # TODO this is a problem
    # transform=transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.1307,), (0.3081,))
    #     ])

    # #TODO We are going to use the test set as validation; is this a problem for personalization? 
    # #TODO Should the test sets be the same for training the inital model and for personalization
    # train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    # test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)


    # clients, groups, train_data, test_data = read_Femnist_data()
    # final_train_data, final_test_data = convert_leafdata_to_emnistform(train_data, test_data)
    ########################################
    ########################################


    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Shakespeare')
    #these arguments do not matter as batch size is created in jupyter notebook
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
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
    parser.add_argument('--cudadevice', type=str, choices=["0", "1", "2", "3"], help='for tracking which device')
    parser.add_argument('--cudabenchmark', action='store_true', default=False,
                        help='quickly check a single pass')

    parser.add_argument('--wandbgroup', type=str, help='wandbgroupname')
    # parser.add_argument('--dropout1', type=float, default=0.4, metavar='DO1',
    #                     help='first dropout rate (default: 0.4)')
    # parser.add_argument('--dropout2', type=float, default=0.4, metavar='DO2',
    #                     help='second dropout rate (default: 0.4)')
    args = parser.parse_args()
    wandb.init(project='shakespeare-P1-PRETRAIN', config=vars(args), group=args.wandbgroup)

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

    # tup = load_partition_data_federated_shakespeare(None, "./FedML/data/fed_shakespeare/datasets")

    # train_loader = tup[3]
    # test_loader = tup[4]


    # with open('shakespeare_data_federated_split.p', "rb") as my_file:
    #     train_loader, valid_loader, test_loader = pickle.load(my_file)
    # #batch size is 128

    train_data_x, train_data_y, val_data_x, val_data_y, test_data_x, test_data_y = torch.load('shakespeare_data_federated_split.pt')

    train_dataset_aug = CustomTensorDataset(tensors=(train_data_x, train_data_y))
    train_loader = torch.utils.data.DataLoader(train_dataset_aug, batch_size=args.batch_size, shuffle=True, pin_memory = True)#, num_workers = 2)

    test_dataset_aug = CustomTensorDataset(tensors=(test_data_x, test_data_y))
    test_loader = torch.utils.data.DataLoader(test_dataset_aug, batch_size=512, shuffle=False, pin_memory = True)#, num_workers = 2)

    val_dataset_aug = CustomTensorDataset(tensors=(val_data_x, val_data_y))
    val_loader = torch.utils.data.DataLoader(val_dataset_aug, batch_size=512, shuffle=False, pin_memory = True)#, num_workers = 2)

    model = RNN_OriginalFedAvg().to(device)
    # torch.save(model, f"UNIT_TEST_MATCHING_BATCH/initialmodel_device={args.cudadevice}.pt")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    wandb.watch(model, log="all")

    for epoch in range(args.epochs):
        train(args, model, device, train_loader, optimizer, epoch, args.cudadevice)
        valid(model, device, val_loader, epoch, args.cudadevice)
        test(model, device, test_loader, epoch, args.cudadevice)
        scheduler.step()

    hyperparameters = [str(args.lr), str(args.gamma)]
    prefix = "_".join(hyperparameters)
    if args.save_model:
        torch.save(model.state_dict(), "shakespeare_P1_pretrained_weights/"+prefix+"_shakespeare_rnn.pt")
        wandb.save("shakespeare_P1_pretrained_weights/"+prefix+'_shakespeare_rnn.pt')

if __name__ == '__main__':
    main()