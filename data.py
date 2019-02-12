import os
import torch

from io import open
from util import load_data

class Dictionary(object):
    # Word level or character level
    def __init__(self):
        self.element2idx = {}
        self.idx2element = []

    def add_element(self, word):
        if word not in self.element2idx:
            self.idx2element.append(word)
            self.element2idx[word] = len(self.idx2element) - 1
        return self.element2idx[word]
    
    def __len__(self):
        return len(self.idx2element)

class Corpus(object):
    def __init__(self, args):
        self.dictionary = Dictionary()

        self.use_char_embeds = args.type == 'char' 
        
        self.train = self.tokenize(os.path.join(args.data, 'train.json'), use_char_embeds=self.use_char_embeds)
        self.valid = self.tokenize(os.path.join(args.data, 'valid.json'), use_char_embeds=self.use_char_embeds)

    def tokenize(self, path, use_char_embeds=False):
        assert os.path.exists(path)

        data = load_data(path)
        data = ' '.join(data['reviewText'])

        
        tokens = 0
        if use_char_embeds:
            data = list(data)
            words = data + ['<eos>']
        else :
             words = data.split() + ['<eos>']

        # Add words to the dictionary
        tokens += len(words)
        for word in words:
            self.dictionary.add_element(word)

        # Tokenbize file content
        ids = torch.LongTensor(tokens)
        token = 0

        for word in words:
                ids[token] = self.dictionary.element2idx[word]
                token += 1
        
        return ids

    def describe(self):

        embed_type = 'char' if self.use_char_embeds else 'word'

        s = f'Using {embed_type} level embeddings\n'
        s += f'Size of dictionary {len(self.dictionary)} \n'
        
        return s