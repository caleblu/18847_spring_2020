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

    def __init__(self,
                 ks,
                 thresholds,
                 kwta,
                 inhibition_radius,
                 rf_size,
                 length,
                 timesteps,
                 inchannels=1):
        super(Column, self).__init__()
        self.k1, self.k2 = ks
        self.thresh1, self.thresh2 = thresholds
        self.kwta = kwta
        self.rf_size = rf_size
        self.inhibition_radius = inhibition_radius
        self.ec1_list = [
            snn.LocalConvolution(input_size=(1, length),
                                 in_channels=inchannels,
                                 out_channels=self.k1,
                                 kernel_size=(1, length),
                                 stride=1) for _ in range(rf_size)
        ]
        self.ec = snn.LocalConvolution(input_size=(rf_size, length),
                                       in_channels=inchannels,
                                       out_channels=self.k2,
                                       kernel_size=(rf_size, length),
                                       stride=1)
        self.ec2 = snn.LocalConvolution(input_size=(1, self.k1 * self.rf_size),
                                        in_channels=inchannels,
                                        out_channels=self.k2,
                                        kernel_size=(1, self.k1 * self.rf_size),
                                        stride=1)

        # self.stdp = snn.ModSTDP(self.ec2,
        #                         10 / 128,
        #                         10 / 128,
        #                         1 / 128,
        #                         96 / 128,
        #                         4 / 128,
        #                         maxweight=timesteps)
        self.stdp = snn.ModSTDP(self.ec,
                                10 / 128,
                                10 / 128,
                                1 / 128,
                                96 / 128,
                                4 / 128,
                                maxweight=timesteps)

    def forward(self, rec_field):
        # One layer forward
        out = self.ec(rec_field)
        # print(out.shape)
        spike, pot = sf.fire(out, self.thresh2, True)
        # print(pot.shape)
        winners = sf.get_k_winners(pot,
                                   kwta=self.kwta,
                                   inhibition_radius=self.inhibition_radius)
        coef = torch.zeros_like(out)
        coef[:, winners, :, :] = 1
        return torch.mul(pot, coef).sign()
        return pot

    # def forward(self, rec_field):
    #     ## 2-layer forward
    #     outs = [
    #         self.ec1_list[i](torch.unsqueeze(rec_field[:, :, i, :], 2))
    #         for i in range(self.rf_size)
    #     ]
    #     # print(outs[0].shape)
    #     pots = torch.cat([sf.fire(o, self.thresh1, True)[1] for o in outs],
    #                      1).squeeze().unsqueeze(1).unsqueeze(1)
    #     # print(pots.shape)
    #     out = self.ec2(pots)
    #     spike, pot = sf.fire(out, self.thresh2, True)
    #     winners = sf.get_k_winners(pot,
    #                                kwta=self.kwta,
    #                                inhibition_radius=self.inhibition_radius)
    #     coef = torch.zeros_like(out)
    #     coef[:, winners, :, :] = 1
    #     return torch.mul(pot, coef).sign()


class DatasetContext(Dataset):

    def __init__(self,
                 words,
                 corpus,
                 spike_input,
                 spike_output,
                 timesteps,
                 transform=None):
        self.words = words
        self.corpus = corpus
        self.spike_input = spike_input
        self.spike_output = spike_output
        self.temporal_transform = utils.Intensity2Latency(timesteps)
        self.data = torch.cat([self.get_context(word) for word in self.words],
                              1)
        self.data_size = self.data.shape[1]

    def __len__(self):
        return self.data.size(1)

    def get_context(self, word):
        idx = self.corpus.dictionary.word2idx[word]
        enc = self.corpus.dictionary.idx2spike[idx]
        context = self.spike_input[np.all(self.spike_output == enc, axis=1)]
        context = torch.from_numpy(context)
        context = self.temporal_transform(context)
        return context.sign()

    def __getitem__(self, index):
        context = torch.unsqueeze(self.data[:, index, :, :], 1)
        return context


def train_stdp(dataset_context, column, num_epochs, batch_size=1000):
    train_loader = DataLoader(dataset_context,
                              batch_size=batch_size,
                              shuffle=True)
    result = torch.zeros(num_epochs * dataset_context.data_size, column.k2)
    for epochs in range(num_epochs):
        start = time.time()
        cnt = 0
        for data in tqdm(train_loader):
            print(data.shape)
            for i in range(len(data)):
                #                 print(i)
                out = column(data[i])
                result[epochs * dataset_context.data_size + i, :] = torch.sum(
                    out.squeeze(), dim=0)
                ## Now stdp only works for 1-layer forward in column
                column.stdp(data[i], out)
        end = time.time()
    print("Training done under ", end - start)

    return result.numpy()


def infer_stdp(dataset_context, column, batch_size=1000):
    train_loader = DataLoader(dataset_context,
                              batch_size=batch_size,
                              shuffle=False)
    result = torch.zeros(dataset_context.data_size, column.k2)
    # print(dataset_context.data_size)
    for data in tqdm(train_loader):
        # print(data.shape)
        for i in range(len(data)):
            #                 print(i)
            out = column(data[i])
            # print(out.shape)
            result[i, :] = torch.sum(out.squeeze(), dim=0)
            ## Now stdp only works for 1-layer forward in column
    # print("Training done under ", end - start)

    return result.numpy()