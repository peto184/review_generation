import torch
import time
import math
import os

from evaluate import evaluate
from inference import inference
from util import get_batch, batchify, repackage_hidden

import tensorflow as tf
import numpy as np

_time = time.strftime('%Y_%m_%d__%H:%M:%S')
LOG_FOLDER = f'./log/{_time}'
summary_writer = tf.summary.FileWriter(LOG_FOLDER)  

def inject_summary(summary_writer, tag, value, step):
    summary = tf.Summary(
        value=[tf.Summary.Value(tag=tag, simple_value=value)])
    summary_writer.add_summary(summary, global_step=step)

def inject_summary_text(summary_writer, tag, value, step):
    t = tf.constant([value])
    summary = tf.summary.text(tag, t)

    with tf.Session() as sess:
        s = sess.run(summary)
        summary_writer.add_summary(s)


def _train_epoch(args, epoch, model, train_data, corpus, device, lr, criterion):
    total_loss = 0.
    start_time = time.time()
    ntokens = len(corpus.dictionary)
    
    hidden = model.init_hidden(args.batch_size)
    
    model.train()
    for batch, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):

        data, targets = get_batch(args.bptt, train_data, i)
        
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        hidden = repackage_hidden(hidden)
        model.zero_grad()
        output, hidden = model(data, hidden)
        loss = criterion(output.view(-1, ntokens), targets)
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        for p in model.parameters():
            p.data.add_(-lr, p.grad.data)

        total_loss += loss.item()

        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, len(train_data) // args.bptt, lr,
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()

            # Log to tensorboard
            info = {
                f'training/{epoch}loss/loss': cur_loss,
                f'training/{epoch}/loss_exp': math.exp(cur_loss),
                'training/lr' : lr,
            }

            for tag, value in info.items():
                inject_summary(summary_writer, tag, value, i)

            summary_writer.flush()
            

def train(args, model, corpus, device, criterion):
    # At any point you can hit Ctrl + C to break out of training early.
    lr = args.lr
    best_val_loss = None  

    train_data = batchify(corpus.train, args.batch_size).to(device)
    valid_data = batchify(corpus.valid, args.batch_size).to(device)

    if not os.path.exists(LOG_FOLDER):
        print(f'Creatnig folder {LOG_FOLDER}')
        os.makedirs(LOG_FOLDER)

    try:
        for epoch in range(1, args.epochs+1):
            epoch_start_time = time.time()
            _train_epoch(args, epoch, model, train_data, corpus, device, lr, criterion)
            val_loss = evaluate(args, valid_data, model, corpus, criterion)
            # val_loss = 0.
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                    'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                            val_loss, math.exp(val_loss)))
            print('-' * 89)
            # Save the model if the validation loss is the best we've seen so far.
            if not best_val_loss or val_loss < best_val_loss:
                with open(args.save, 'wb') as f:
                    torch.save(model, f)
                best_val_loss = val_loss
            else:
                # Anneal the learning rate if no improvement has been seen in the validation dataset.
                lr /= 4.0

            # Persist after each epoch
            with open(f"./models/m_{args.type}_{epoch}.pkl", 'wb') as f:
                torch.save(model, f)

            # Inference a text after each epoch
            inference_text = inference(args, model, corpus, device)
            print(f'Generated text: {inference_text}')
            inject_summary_text(summary_writer, f'Inference_{epoch}', inference_text, epoch)

            # Log to tensorboard
            info = {
                'validation/loss/val_loss': val_loss,
                'validation/loss/val_loss_exp': math.exp(val_loss),
            }

            for tag, value in info.items():
                inject_summary(summary_writer, tag, value, epoch)

            summary_writer.flush()
                

    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')