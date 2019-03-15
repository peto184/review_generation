import torch
from util import repackage_hidden, get_batch

temperature = 1.0  # temperature - higher will increase diversity


def inference(args, model, corpus, device):
    # Turn on evaluation mode which disables dropout.
    model.eval()

    hidden = model.init_hidden(1)
    input = torch.randint(len(corpus.dictionary), (1, 1),
                          dtype=torch.long).to(device)

    result = ''

    with torch.no_grad():  # no tracking history
        for i in range(100):
            output, hidden = model(input, hidden)
            element_weights = output.squeeze().div(temperature).exp().cpu()
            element_idx = torch.multinomial(element_weights, 1)[0]
            input.fill_(element_idx)
            word = corpus.dictionary.idx2element[element_idx]

            if args.type == 'word':
                result += word + ('\n' if i % 20 == 19 else ' ')
            else:
                result += word + ('\n' if i % 80 == 79 else '')

    return result
