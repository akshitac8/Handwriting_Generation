import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import nltk
import random
import numpy as np
from collections import Counter, OrderedDict
import nltk
from copy import deepcopy
import os
import re
import io
import unicodedata
import handwriting.models as m

print('Imported')
flatten = lambda l: [item for sublist in l for item in sublist]

from torch.nn.utils.rnn import PackedSequence,pack_padded_sequence
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
random.seed(1024)

USE_CUDA = torch.cuda.is_available()
gpus = [0]
torch.cuda.set_device(gpus[0])

FloatTensor = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if USE_CUDA else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if USE_CUDA else torch.ByteTensor

def getBatch(batch_size, train_data):
    random.shuffle(train_data)
    sindex=0
    eindex=batch_size
    while eindex < len(train_data):
        batch = train_data[sindex: eindex]
        temp = eindex
        eindex = eindex + batch_size
        sindex = temp
        yield batch
    
    if eindex >= len(train_data):
        batch = train_data[sindex:]
        yield batch
        
# It is for Sequence 2 Sequence format
def pad_to_batch(batch, x_to_ix, y_to_ix):
    
    sorted_batch =  sorted(batch, key=lambda b:b[0].size(1), reverse=True) # sort by len
    x,y = list(zip(*sorted_batch))
    max_x = max([s.size(1) for s in x])
    max_y = max([s.size(1) for s in y])
    x_p, y_p = [], []
    for i in range(len(batch)):
        if x[i].size(1) < max_x:
            x_p.append(torch.cat([x[i], Variable(LongTensor([x_to_ix['<PAD>']] * (max_x - x[i].size(1)))).view(1, -1)], 1))
        else:
            x_p.append(x[i])
        if y[i].size(1) < max_y:
            y_p.append(torch.cat([y[i], Variable(LongTensor([y_to_ix['<PAD>']] * (max_y - y[i].size(1)))).view(1, -1)], 1))
        else:
            y_p.append(y[i])
        
    input_var = torch.cat(x_p)
    target_var = torch.cat(y_p)
    input_len = [list(map(lambda s: s ==0, t.data)).count(False) for t in input_var]
    target_len = [list(map(lambda s: s ==0, t.data)).count(False) for t in target_var]
    
    return input_var, target_var, input_len, target_len

def prepare_sequence(seq, to_index):
    idxs = list(map(lambda w: to_index[w] if to_index.get(w) is not None else to_index["<UNK>"], seq))
    return Variable(LongTensor(idxs))

# Turn a Unicode string to plain ASCII, thanks to http://stackoverflow.com/a/518232/2809427
def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters
def normalize_string(s):
    s = unicode_to_ascii(s.lower().strip())
    s = re.sub(r"([,.!?])", r" \1 ", s)
    s = re.sub(r"[^a-zA-Z,.!?]+", r" ", s)
    s = re.sub(r"\s+", r" ", s).strip()
    return s

texts = io.open('data/sentences.txt', 'r', encoding='utf-8').readlines()
#with io.open('data/sentences.txt','r') as corpus:
#    texts = corpus.readlines()
    #texts = corpus.read().strip().split('.')
strokes = np.load('data/strokes.npy', encoding="latin1")
#print(len(strokes))
#print(texts[2])
#print(len(texts))
texts = texts[:1000]

MIN_LENGTH = 5
MAX_LENGTH = 25

X_r, y_r = [], [] # raw
for parallel in texts:
    so,ta = strokes, texts
    for ta in texts:
        if len(ta) < 1: 
            continue
        #print(ta)
        #normalized_so = normalize_string(so).split()
        #print(so)
        normalized_ta = normalize_string(ta).split()
        
        #print(normalized_ta)
        
        if len(normalized_ta) >= MIN_LENGTH and len(normalized_ta) <= MAX_LENGTH:
            X_r.append(so)
            #print('di') 
            y_r.append(normalized_ta)
    

print(len(X_r), len(y_r))
print("content")
print(X_r[0], y_r[0])
#X_r = np.ndarray(X_r)
#y_r = np.ndarray(y_r)
print(type(X_r))
print(type(y_r)) 
#X_r = X_r.tolist()
#y_r = y_r.tolist()
source_vocab = X_r
target_vocab = y_r

#X_r= set(X_r)

#source_vocab = list(X_r)
#target_vocab = list(y_r)
print(len(source_vocab), len(target_vocab))
#X_r = hash(X_r)
#y_r = hash(y_r)

#source2index = {'<PAD>': 0, '<UNK>': 1, '<s>': 2, '</s>': 3}
#for vo in source_vocab:
#    if source2index.get(vo) is None:
#        source2index[vo] = len(source2index)
#index2source = {v:k for k, v in source2index.items()}

target2index = {'<PAD>': 0, '<UNK>': 1, '<s>': 2, '</s>': 3}
for vo in target_vocab:
    if target2index.get(vo) is None:
        target2index[vo] = len(target2index)
index2target = {v:k for k, v in target2index.items()}

X_p, y_p = [], []

for so, ta in zip(X_r, y_r):
    #X_p.append(prepare_sequence(so + ['</s>'], source2index).view(1, -1))
    X_p.append(prepare_sequence(source_vocab).view(1, -1))
    y_p.append(prepare_sequence(ta + ['</s>'], target2index).view(1, -1))

print("data")
print(X_p)
print(y_p)    
train_data = list(zip(X_p, y_p))

EPOCH = 50
BATCH_SIZE = 64
EMBEDDING_SIZE = 300
HIDDEN_SIZE = 512
LR = 0.001
DECODER_LEARNING_RATIO = 5.0
RESCHEDULED = False

bre

encoder = m.Encoder(len(source_vocab), EMBEDDING_SIZE, HIDDEN_SIZE, 3, True)
decoder = m.Decoder(len(target2index), EMBEDDING_SIZE, HIDDEN_SIZE * 2)
encoder.init_weight()
decoder.init_weight()

if USE_CUDA:
    encoder = encoder.cuda()
    decoder = decoder.cuda()

loss_function = nn.CrossEntropyLoss(ignore_index=0)
enc_optimizer = optim.Adam(encoder.parameters(), lr=LR)
dec_optimizer = optim.Adam(decoder.parameters(), lr=LR * DECODER_LEARNING_RATIO)

for epoch in range(EPOCH):
    losses=[]
    for i, batch in enumerate(getBatch(BATCH_SIZE, train_data)):
        inputs, targets, input_lengths, target_lengths = pad_to_batch(batch, source_vocab, target2index)
        
        input_masks = torch.cat([Variable(ByteTensor(tuple(map(lambda s: s ==0, t.data)))) for t in inputs]).view(inputs.size(0), -1)
        start_decode = Variable(LongTensor([[target2index['<s>']] * targets.size(0)])).transpose(0, 1)
        encoder.zero_grad()
        decoder.zero_grad()
        output, hidden_c = encoder(inputs, input_lengths)
        
        preds = decoder(start_decode, hidden_c, targets.size(1), output, input_masks, True)
                                
        loss = loss_function(preds, targets.view(-1))
        losses.append(loss.data.tolist() )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), 50.0) # gradient clipping
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), 50.0) # gradient clipping
        enc_optimizer.step()
        dec_optimizer.step()

        if i % 200==0:
            print("[%02d/%d] [%03d/%d] mean_loss : %0.2f" %(epoch, EPOCH, i, len(train_data)//BATCH_SIZE, np.mean(losses)))
            losses=[]

    # You can use http://pytorch.org/docs/master/optim.html#how-to-adjust-learning-rate
    if RESCHEDULED == False and epoch  == EPOCH//2:
        LR *= 0.01
        enc_optimizer = optim.Adam(encoder.parameters(), lr=LR)
        dec_optimizer = optim.Adam(decoder.parameters(), lr=LR * DECODER_LEARNING_RATIO)
        RESCHEDULED = True
        

# borrowed code from https://github.com/spro/practical-pytorch/blob/master/seq2seq-translation/seq2seq-translation-batched.ipynb

def show_attention(input_words, output_words, attentions):
    # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.numpy(), cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + input_words, rotation=90)
    ax.set_yticklabels([''] + output_words)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

#     show_plot_visdom()
    plt.show()
    plt.close()
    

test = random.choice(train_data)
input_ = test[0]
truth = test[1]

output, hidden = encoder(input_, [input_.size(1)])
pred, attn = decoder.decode(hidden, output)

input_ = [index2source[i] for i in input_.data.tolist()[0]]
pred = [index2target[i] for i in pred.data.tolist()]


print('Source : ',' '.join([i for i in input_ if i not in ['</s>']]))
print('Truth : ',' '.join([index2target[i] for i in truth.data.tolist()[0] if i not in [2, 3]]))
print('Prediction : ',' '.join([i for i in pred if i not in ['</s>']]))

if USE_CUDA:
    attn = attn.cpu()

show_attention(input_, pred, attn.data)