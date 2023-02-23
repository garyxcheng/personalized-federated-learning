import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Last_Layer_CIFAR100_Net(nn.Module):
    """
    This is the last layer from Resnet18 from personalizedFL/preprocess_and_pretrain/cutout_resnet.py 
    """
    def __init__(self):
        num_classes = 100
        block_expansion = 1 #see lines 145 and 73 of resnet.py
        super(Last_Layer_CIFAR100_Net, self).__init__()
        self.linear = nn.Linear(512 * block_expansion, num_classes)
    
    def forward(self, x):
        x = self.linear(x)
        output = F.log_softmax(x, dim=1)
        return output

class Last_Layer_Shakespeare_Net(nn.Module):

    def __init__(self, embedding_dim=8, vocab_size=90, hidden_size=256):
        super(Last_Layer_Shakespeare_Net, self).__init__()
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        # the input x here is the output (of the penultimate layer) from a pretrained net. 
        # We have stored the output of the pretrained net

        # For fed_shakespeare
        output = self.fc(x)
        output = torch.transpose(output, 1, 2)

        #### We add this to make it work for our setup
        output = F.log_softmax(output, dim=1)
        ####

        return output


class Last_Layer_Stackoverflownwp_Net(nn.Module):

    def __init__(self, vocab_size=10000,
                 num_oov_buckets=1,
                 embedding_size=96,
                 latent_size=670,
                 num_layers=1):
        super(Last_Layer_Stackoverflownwp_Net, self).__init__()
        extended_vocab_size = vocab_size + 3 + num_oov_buckets  # For pad/bos/eos/oov.
        # self.word_embeddings = nn.Embedding(num_embeddings=extended_vocab_size, embedding_dim=embedding_size,
        #                                     padding_idx=0)
        # self.lstm = nn.LSTM(input_size=embedding_size, hidden_size=latent_size, num_layers=num_layers)
        # self.fc1 = nn.Linear(latent_size, embedding_size)
        self.fc2 = nn.Linear(embedding_size, extended_vocab_size)

    # def forward(self, input_seq, hidden_state = None):
    def forward(self, x):
        # embeds = self.word_embeddings(input_seq)
        # lstm_out, hidden_state = self.lstm(embeds, hidden_state)
        # fc1_output = self.fc1(lstm_out[:,:])
        output = self.fc2(x)
        output = torch.transpose(output, 1, 2)

        #### We add this to make it work for our setup
        #TODO: check to make sure this is correct
        output = F.log_softmax(output, dim=1)

        return output

class Last_Layer_EMNIST_Net(nn.Module):
    def __init__(self):
        super(Last_Layer_EMNIST_Net, self).__init__()
        self.fc2 = nn.Linear(512, 62)

    def forward(self, x):
        # the input x here is the output (of the penultimate layer) from a pretrained net. 
        # We have stored the output of the pretrained net

        # x = torch.transpose(torch.reshape(x, (-1, 1, 28, 28)), 2, 3) #TODO fix this issue
        x = torch.reshape(x, (-1, 512)) #TODO fix this issue
        
        x = F.relu(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output
