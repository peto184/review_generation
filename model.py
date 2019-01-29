import torch.nn as nn


class RNNModel(nn.Module):
    def __init__(self, ntoken, ninp, nhid, nlayers, dropout=0.5, tie_weigths=False):
        super(RNNModel, self).__init__()

        self.encoder = nn.Embedding(ntoken, ninp)
        self.drop = nn.Dropout(dropout)
        self.rnn = nn.LSTM(ninp, nhid, nlayers, dropout=dropout)
        self.decoder = nn.Linear(nhid, ntoken)

        if tie_weigths:
            if nhid != ninp:
                raise ValueError(
                    "When using tied flag, nhid must be equal to embeddings size.")
            self.decoder.weights = self.encoder.weights

        self.init_weights()

        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initRange = 0.1

        self.encoder.weight.data.uniform_(-initRange, initRange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initRange, initRange)

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                weight.new_zeros(self.nlayers, bsz, self.nhid))

    def forward(self, input, hidden):
        emb = self.drop(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        dec = self.decoder(output.view(
            output.size(0) * output.size(1), output.size(2)))

        return dec.view(output.size(0), output.size(1), dec.size(1)), hidden
