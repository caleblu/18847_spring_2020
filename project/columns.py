import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.nn.parameter import Parameter
import torchvision
from torchvision import transforms

from SpykeTorch import snn
from SpykeTorch import functional as sf
from SpykeTorch import visualization as vis
from SpykeTorch import utils
import numpy as np
import time
from tqdm import tqdm


class Column(nn.Module):

    def __init__(
            self,
            k,  #number of output channels 
            threshold,
            kwta,
            inhibition_radius,
            rf_size,  #number of context words
            length,  #length of 1st layer encoding = number of output channels
            timesteps,
            inchannels=1):

        super(Column, self).__init__()
        self.k = k
        self.thresh = threshold
        self.kwta = kwta
        self.inhibition_radius = inhibition_radius
        self.ec = snn.LocalConvolution(input_size=(rf_size, length),
                                       in_channels=inchannels,
                                       out_channels=self.k,
                                       kernel_size=(length, rf_size),
                                       stride=1)
        #ucapture, uminus, usearch, ubackoff, umin, maxweight
        self.rstdp = snn.ModRSTDP(self.ec,
                                  4 / 128,
                                  4 / 128,
                                  2 / 128,
                                  94 / 128,
                                  4 / 128,
                                  maxweight=timesteps)

    def forward(self, rec_field):
        ###############################################################################
        #output = Column(rec_field)
        #input:
        #       rec_field: first-layer output       [timesteps, 1, k, rf_size]
        #output:
        #       center word representation          [timesteps, k, 1, 1]
        ###############################################################################
        #output size: (timesteps, k, 1, 1)
        out = self.ec(rec_field)
        spike, pot = sf.fire(out, self.thresh, True)
        winners = sf.get_k_winners(pot,
                                   kwta=self.kwta,
                                   inhibition_radius=self.inhibition_radius)
        coef = torch.zeros_like(out)
        coef[:, winners, :, :] = 1
        return torch.mul(pot, coef).sign()


class Column1(nn.Module):

    def __init__(
            self,
            k,  #number of output channels  = length of the encoding
            threshold,
            kwta,
            inhibition_radius,
            rf_size,  #number of context words
            length,  #length of the first layer fixed input encoding
            timesteps,
            inchannels=1):
        super(Column1, self).__init__()
        self.k = k
        self.thresh = threshold
        self.kwta = kwta
        self.rf_size = rf_size
        self.inhibition_radius = inhibition_radius

        self.ec_list = [
            snn.LocalConvolution(input_size=(1, length),
                                 in_channels=inchannels,
                                 out_channels=self.k,
                                 kernel_size=(1, length),
                                 stride=1) for _ in range(rf_size)
        ]
        self.rstdp_list = [
            snn.ModRSTDP(self.ec_list[i],
                         10 / 128,
                         9 / 128,
                         4 / 128,
                         100 / 128,
                         4 / 128,
                         maxweight=timesteps) for i in range(rf_size)
        ]

    def forward(self, rec_field, reward):
        ###############################################################################
        #output = Column1(rec_field, reward)
        #input:
        #       rec_field: fixed input encoding     [timesteps, 1, rf_size, length]
        #       reward: context word reward         [timesteps, k, rf_size]
        #output:
        #       context word 1st layer output      [timesteps, 1, k, rf_size]
        ###############################################################################
        outs = [
            self.ec_list[i](torch.unsqueeze(rec_field[:, :, i, :], 2))
            for i in range(self.rf_size)
        ]

        pots = [sf.fire(o, self.thresh, True)[1] for o in outs]

        #kwta
        winners = [
            sf.get_k_winners(pot,
                             kwta=self.kwta,
                             inhibition_radius=self.inhibition_radius)
            for pot in pots
        ]
        coefs = [torch.zeros_like(out) for out in outs]
        for i in range(self.rf_size):
            coefs[i][:, winners[i], :, :] = 1
        pots = [
            torch.mul(pots[i], coefs[i]).sign() for i in range(self.rf_size)
        ]
        pots = torch.cat(pots, 2).squeeze().unsqueeze(1)

        #feedback
        for i in range(self.rf_size):
            self.rstdp_list[i](torch.unsqueeze(rec_field[:, :, i, :],
                                               2), pots[:, :, :, i].squeeze(),
                               reward[:, :, i].squeeze())

        return pots


##############################################################
# word                  'dog'
# index                 3
# encode                [1 x length]
# temperol spikes       [timesteps x length] --> {0,1}
# SynDataset.__getitem__(i)
# output: (context_spikes, context_index, c_ward_index)
#           context_spikes:  [timesteps x 1 x #context words (rf_field) x length] --> {0,1}
#           context_index:   [#context words (rf_field) x 1]
#           c_word_index:    center word index
##############################################################


class SynDataset(Dataset):

    def __init__(
            self,
            corpus,
            spike_input,  #the encoding of the context words for all the moving windows. [#windows x #context words (rf_field) x length]
            input,  #the index of the context words for all the moving windows  [#windows x #context words (rf_field)]
            output,  #the index of the center word for all the moving windows  [#windows]
            timesteps,
            words=None,  # an array of words as center words if set, select windows with center words in 'words'
            transform=None):

        self.corpus = corpus
        self.spike_input = spike_input
        self.input = np.array(input)
        self.output = np.array(output)
        self.words = words
        self.temporal_transform = utils.Intensity2Latency(timesteps)
        if words is not None:
            context_spike = []
            context_v = []
            out_v = []
            for word in self.words:
                spike, v, center_v = self.get_context(word)
                context_spike.append(spike)
                context_v.append(v)
                out_v.append(center_v)
            self.spike_input = np.concatenate(context_spike)
            self.input = np.concatenate(context_v)
            self.output = np.concatenate(out_v)
        self.data_size = len(self.output)

    def __len__(self):
        return len(self.output)

    def get_context(self, word):
        idx = self.corpus.dictionary.word2idx[word]
        ind = np.argwhere(self.output == idx)[:, 0]
        context_spike = self.spike_input[ind, :, :]
        context_v = self.input[ind, :]
        out_v = self.output[ind]
        return context_spike, context_v, out_v

    def __getitem__(self, index):
        context = torch.from_numpy(self.spike_input[index, :, :])
        return self.temporal_transform(context).sign().unsqueeze(
            1), self.input[index], self.output[index]


##############################################################
# input:
#           dataset: SynDataset
#           column1: Column1 (first layer)
#           column2: Column (second layer)
#           vec_length: k --> length of the final encoding
#           num_epochs: number of epochs
#           R: learned representation (2nd layer output)  [timesteps,  k,  vocabulary size]
# output: (result, result_label, R)
#           result:  np.array [epoches*data_size x k]
#           result_label:   the center word index for each training
#           R:    updated learned representation [timesteps,  k,  vocabulary size] (only those words in result_label are updated)
##############################################################


def train_rstdp(dataset, column1, column2, vec_length, num_epochs, R):
    train_loader = DataLoader(dataset,
                              batch_size=dataset.data_size,
                              shuffle=True)
    result = torch.zeros(num_epochs * dataset.data_size, vec_length)
    result_label = np.zeros((num_epochs * dataset.data_size))
    for epochs in range(num_epochs):
        start = time.time()
        cnt = 0
        for input_temp, input_r, output_r in tqdm(train_loader):
            input_r = input_r.type(torch.int64)
            output_r = output_r.type(torch.int64)
            for i in range(len(input_temp)):
                #first layer and feedback R[context_words]
                #print(input_r[i], output_r[i])
                for j in range(5):
                    out = column1(input_temp[i], R[:, :, input_r[i]])
                    #second layer
                    out2 = column2(out)
                    #second layer feedback R[center_word]
                    column2.rstdp(out, out2, R[:, :, output_r[i]])
                    #update R[center_word]
                    R[:, :, output_r[i]] = out2.squeeze()
                    #record second layer output and the corresponding center word index
                    result[epochs * dataset.data_size + i, :] = torch.sum(
                        out2.squeeze(), dim=0)
                result_label[epochs * dataset.data_size + i] = output_r[i]
        end = time.time()
    print("Training done under ", end - start)
    return result.numpy(), result_label, R


# class DatasetContext(Dataset):
#     def __init__(self,
#                  words,
#                  corpus,
#                  spike_input,
#                  spike_output,
#                  timesteps,
#                  transform=None):
#         self.words = words
#         self.corpus = corpus
#         self.spike_input = spike_input
#         self.spike_output = spike_output
#         self.temporal_transform = utils.Intensity2Latency(timesteps)
#         self.data = torch.cat([self.get_context(word) for word in self.words],
#                               1)
#         self.data_size = self.data.shape[1]

#     def __len__(self):
#         return self.data.size(1)

#     def get_context(self, word):
#         idx = self.corpus.dictionary.word2idx[word]
#         enc = self.corpus.dictionary.idx2spike[idx]
#         context = self.spike_input[np.all(self.spike_output == enc, axis=1)]
#         context = torch.from_numpy(context)
#         context = self.temporal_transform(context)
#         return context.sign()

#     def __getitem__(self, index):
#         context = torch.unsqueeze(self.data[:, index, :, :], 1)
#         return context


# def train_stdp(dataset_context, column, num_epochs, batch_size=1000):
#     train_loader = DataLoader(dataset_context,
#                               batch_size=batch_size,
#                               shuffle=True)
#     result = torch.zeros(num_epochs * dataset_context.data_size, column.k2)
#     for epochs in range(num_epochs):
#         start = time.time()
#         cnt = 0
#         for data in tqdm(train_loader):
#             print(data.shape)
#             for i in range(len(data)):
#                 #                 print(i)
#                 out = column(data[i])
#                 result[epochs * dataset_context.data_size + i, :] = torch.sum(
#                     out.squeeze(), dim=0)
#                 ## Now stdp only works for 1-layer forward in column
#                 column.stdp(data[i], out)
#         end = time.time()
#     print("Training done under ", end - start)

#     return result.numpy()
# def infer_stdp(dataset_context, column, batch_size=1000):
#     train_loader = DataLoader(dataset_context,
#                               batch_size=batch_size,
#                               shuffle=False)
#     result = torch.zeros(dataset_context.data_size, column.k2)
#     # print(dataset_context.data_size)
#     for data in tqdm(train_loader):
#         # print(data.shape)
#         for i in range(len(data)):
#             #                 print(i)
#             out = column(data[i])
#             # print(out.shape)
#             result[i, :] = torch.sum(out.squeeze(), dim=0)
#             ## Now stdp only works for 1-layer forward in column
#     # print("Training done under ", end - start)

#     return result.numpy()
