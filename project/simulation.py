import itertools
import os


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


class Corpus(object):

    def __init__(self):
        self.dictionary = Dictionary()

    def tokenize(self, sentences):

        for line in sentences:
            words = line.split() + ['<eos>']
            for word in words:
                self.dictionary.add_word(word)

        # Tokenize file content
        idss = []
        for line in sentences:
            words = line.split() + ['<eos>']
            ids = []
            for word in words:
                ids.append(self.dictionary.word2idx[word])
            idss.append(ids)

        return idss


class Simulation():

    def __init__(self):
        self.terminals = {
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
            'EOS': ['.']
        }

    def construct_sentences(self):
        combs = itertools.product(*self.terminals.values())
        return [' '.join(s) for s in list(combs)]
