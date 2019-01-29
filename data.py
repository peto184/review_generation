import os
import torch

from io import open
from util import load_data

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
    def __init__(self, args):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(args.data, 'train.json'))
        self.valid = self.tokenize(os.path.join(args.data, 'valid.json'))

    def tokenize(self, path):
        assert os.path.exists(path)

        data = load_data(path)
        data = ' '.join(data['reviewText'])

        # Add words to the dictionary
        tokens = 0
        words = data.split() + ['<eos>']    
        tokens += len(words)
        for word in words:
            self.dictionary.add_word(word)

        # Tokenbize file content
        ids = torch.LongTensor(tokens)
        token = 0

        for word in words:
                ids[token] = self.dictionary.word2idx[word]
                token += 1
        
        return ids