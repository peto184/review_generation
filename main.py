import pandas as pd
import numpy as np
import data
import argparse
import torch

from util import load_data, batchify
from train import train
from model import RNNModel

parser = argparse.ArgumentParser(description='PyTorch implementation of text generation with LSTM')
parser.add_argument('--data', type=str, default='./data/',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')
parser.add_argument('--type', type=str, default='word',
                    help='Whether to use character or word level embedding. (word|char)')

parser.add_argument('--emsize', type=int, default=200,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=200,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=2,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=20,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=40,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=35,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights')

parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str, default='./output/model.pt',
                    help='path to save the final model')
parser.add_argument('--onnx-export', type=str, default='./output/',
                    help='path to export the final model in onnx format')
args = parser.parse_args()

# Load Data

device = torch.device("cuda" if args.cuda else "cpu")
print(f"Using {device} to train.")

print('Loading corpus.')
corpus = data.Corpus(args)

print(corpus.describe())
ntokens = len(corpus.dictionary)
model = RNNModel(ntokens, args.emsize, args.nhid, args.nlayers, dropout=args.dropout).to(device)
criterion = torch.nn.CrossEntropyLoss()

train(args, model, corpus, device, criterion)

# Build the model

