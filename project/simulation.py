import itertools
import os
from itertools import combinations_with_replacement, combinations, permutations
import numpy as np
import copy


def center_gaussian(i, length, nuerons):
    return (2 * i - 3) / 2 * length / (nuerons - 2)


def gaussian(a, x, mu, sig):
    return a - (a * np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.))))


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)

    def get_encoding(self, length, spike):
        comb = [
            list(permutations(p)) for p in list(
                combinations_with_replacement(np.arange(spike), length))
        ]
        self.idx2spike = np.array(
            list(set([item for sublist in comb for item in sublist
                     ]))[:len(self.word2idx)])
        print("idx2spike: ", self.idx2spike.shape)
        print("idx2spike: ", self.idx2spike)

    def get_gaussian_encoding(self, word, neurons, spike, sig):
        self.idx2spike = np.zeros((word, neurons))
        for x in range(word):
            for y in range(neurons):
                #self.idx2spike[x,0] = spike
                #self.idx2spike[x,neurons-1] = spike
                self.idx2spike[x, y] = round(
                    gaussian(spike, x, center_gaussian(y, spike, neurons), sig))
        print("idx2spike: ", self.idx2spike.shape)
        print("idx2spike: ", self.idx2spike)


class SpikeData(object):
    def __init__(self, tokens, sentences, corpus):
        self.sentences = sentences
        self.tokens = tokens
        self.corpus = corpus
        self.length = len(self.tokens[0])
        self.data_size = len(self.tokens)

    def convert_tokens(self, window_size):
        spike_input = []
        spike_output = []
        input = []
        output = []
        ws = 2 * window_size + 1
        for t in self.tokens:
            for i in range(self.length - ws + 1):
                spike_input.append(
                    self.corpus.dictionary.idx2spike.take(np.hstack(
                        (t[i:window_size + i],
                         t[window_size + 1 + i:window_size + 1 + i +
                           window_size])),
                                                          axis=0))
                spike_output.append(
                    self.corpus.dictionary.idx2spike.take(t[window_size + i],
                                                          axis=0))
                output.append(t[window_size + i])
                input.append(np.concatenate((t[i:window_size + i],t[window_size+1+i:window_size + 1 + i +window_size])))
        spike_input = np.array(spike_input)
        spike_output = np.array(spike_output)
        return spike_input, input, spike_output, output


class Corpus(object):

    def __init__(self):
        self.dictionary = Dictionary()

    def tokenize(self, sentences):

        for line in sentences:
            words = line.split()
            for word in words:
                self.dictionary.add_word(word)

        # Tokenize file content
        idss = []
        for line in sentences:
            words = line.split()
            ids = []
            for word in words:
                ids.append(self.dictionary.word2idx[word])
            idss.append(ids)

        return np.array(idss)


class Simulation():

    def __init__(self):
        self.sentence_order = [
            'S1', 'S2', 'D', 'SUBJ', 'MV', 'IP', 'D1', 'LOC', 'PERIOD', 'E1',
            'E2'
        ]
        self.terminals = {
            'S1': ['<sos>'],
            'S2': ['<sos>'],
            'D': ['the'],
            'SUBJ': [
                'cat', 'dog', 'cow', 'sheep', 'goat', 'pig', 'duck', 'chicken',
                'bird', 'poodle'
            ],
            'MV': [
                'sits', 'eats', 'drinks', 'rests', 'sleeps', 'laughs', 'plays',
                'is', 'lives', 'waits'
            ],
            'IP': ['on', 'by', 'near', 'behind'],
            'D1': ['the'],
            'LOC': [
                'table', 'chair', 'sofa', 'mat', 'tv', 'dresser', 'shelf',
                'couch', 'desk', 'lamp', 'computer', 'keyboards'
            ],
            'PERIOD': ['.'],
            'E1': ['<eos>'],
            'E2': ['<eos>']
        }
        self.special_terminals = {
            w: self.terminals[w]
            for w in ['S1', 'S2', 'D', 'IP', 'D1', 'LOC', 'PERIOD', 'E1', 'E2']
        }
        self.cat_terminals = copy.deepcopy(self.special_terminals)
        self.cat_terminals['SUBJ'] = ['cat']
        self.cat_terminals['MV'] = ['purrs']

        self.dog_terminals = copy.deepcopy(self.special_terminals)
        self.dog_terminals['SUBJ'] = ['dog', 'poodle']
        self.dog_terminals['MV'] = ['barks']

    def construct_sentences(self):

        def get_terminal_values(terminal):
            return [terminal[w] for w in self.sentence_order]

        combs = itertools.product(*get_terminal_values(self.terminals))
        combs_cat = itertools.product(*get_terminal_values(self.cat_terminals))
        combs_dog = itertools.product(*get_terminal_values(self.dog_terminals))

        return [
            ' '.join(s) for s in list(combs) + list(combs_cat) + list(combs_dog)
        ]
