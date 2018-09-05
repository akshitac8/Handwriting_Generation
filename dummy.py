import os
import torch
import sys
sys.path.append('/Users/agupta/version-control/pytorch/Handwriting_Generation')
import numpy as np
from tqdm import tqdm
from collections import defaultdict,namedtuple
import src.models as m
import src.inference_utils as i_utils
from utils import visualization_utils as v_utils
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import configurations

def process_text_data(conf, texts):
    print ('using process_text_data function')

    texts = [l.rstrip()
              .replace("!", "#")
              .replace("?", "#")
              .replace(":", "#")
              .replace(",", "#")
              .replace(".", "#")
              .replace(";", "#")
              .replace("(", "#")
              .replace(")", "#")
              .replace("#", "#")
              .replace("\'", "#")
              .replace("\"", "#")
              .replace("+", "#")
              .replace("-", "#")
              .replace("/", "#")
              .replace("0", "#")
              .replace("1", "#")
              .replace("2", "#")
              .replace("3", "#")
              .replace("4", "#")
              .replace("5", "#")
              .replace("6", "#")
              .replace("7", "#")
              .replace("8", "#")
              .replace("9", "#")
             for l in texts]

    # Get list of unique characters
    conf.alphabet = np.unique(list("".join(texts))).tolist()
    conf.n_alphabet = len(conf.alphabet)

    # Dict mapping unique characters to an index and vice versa
    conf.d_char_to_idx = {}
    conf.d_idx_to_char = {}
    for char_idx, char in enumerate(conf.alphabet):
        conf.d_char_to_idx[char] = char_idx
        conf.d_idx_to_char[char_idx] = char

    # One hot encode the sequences
    texts_one_hot = []
    for line in texts:
        # Split line into its individual characters
        line_chars = list(line)
        one_hot = np.zeros((len(line_chars), conf.n_alphabet), dtype=np.float32)
        # Fill the one hot encoding
        for i, char in enumerate(line_chars):
            one_hot[i, conf.d_char_to_idx[char]] = 1.0
        texts_one_hot.append(one_hot)

    #print(conf)
    #time.sleep(1000)

    return conf, texts, texts_one_hot


def load_data(conf, data_path="data"):
    # Load the array of strokes
    raw_strokes = np.load('/Users/agupta/version-control/pytorch/Handwriting_Generation/data/strokes.npy' , encoding="latin1")
    # Load the list of sentences
    with open('/Users/agupta/version-control/pytorch/Handwriting_Generation/data/sentences.txt' ) as f:
        raw_texts = f.readlines()

    # We will compute the mean ratio len_stroke / len_onehot
    stroke_counter, text_counter = 0, 0
    # We remove pairs of (stroke, text) where len(stroke) < conf.bptt
    strokes, texts = [], []
    for s, t in zip(raw_strokes, raw_texts):
        if s.shape[0] > conf.bptt + 1:
            strokes.append(s)
            texts.append(t)
            # Update our stroke and text counters
            stroke_counter += s.shape[0]
            text_counter += len(t)

    # Compute the mean ratio len_stroke / len_onehot (used in conditional generation)
    conf.stroke_onehot_ratio = int(stroke_counter / text_counter)

    
    conf, texts, onehots = process_text_data(conf, texts)
   
    rng_state = np.random.get_state()
    np.random.shuffle(strokes)
    #print(strokes.type())  
    np.random.set_state(rng_state)
    np.random.shuffle(texts)
    np.random.set_state(rng_state)
    np.random.shuffle(onehots)

        
    return strokes, texts, onehots


def load_uncond_data(conf, data_path="data"):

    raw_strokes = np.load('/Users/agupta/version-control/pytorch/Handwriting_Generation/data/strokes.npy' , encoding="latin1")
    with open('/Users/agupta/version-control/pytorch/Handwriting_Generation/data/sentences.txt' ) as f:
        raw_texts = f.readlines()

    stroke_counter, text_counter = 0, 0
    strokes, texts = [], []
    for s, t in zip(raw_strokes, raw_texts):
        if s.shape[0] > conf.bptt + 1:
            strokes.append(s)
            texts.append(t)
            # Update our stroke and text counters
            stroke_counter += s.shape[0]
            text_counter += len(t)

    conf.stroke_onehot_ratio = int(stroke_counter / text_counter)
    rng_state = np.random.get_state()
    np.random.shuffle(strokes)
    
    return strokes

def generate_unconditionally():
    # Input:
    #   random_seed - integer

    # Output:
    #   stroke - numpy 2D-array (T x 3)
    settings = configurations.get_args()

    load_uncond_data(settings)

    if not os.path.isfile("/Users/agupta/version-control/pytorch/Handwriting_Generation/models/unconditional.pt"):
        print("Unconditional model does not exist.")

    # Load model
    model = torch.load("/Users/agupta/version-control/pytorch/Handwriting_Generation/models/unconditional.pt")
    
    print("model loaded")
    # Sample a sequence to follow progress and save the plot
    plot_data = i_utils.getuncond_sequence(settings, model)

    return plot_data


def generate_conditionally(text="an input string"):
    # Input:
    #   text - str
    #   random_seed - integer

    # Output:
    #   stroke - numpy 2D-array (T x 3)
    settings = configurations.get_args()

    load_data(settings)

    if not os.path.isfile("/Users/agupta/version-control/pytorch/Handwriting_Generation/models/conditional.pt"):
        print("Conditional model does not exist.")

    # Load model
    model = torch.load("/Users/agupta/version-control/pytorch/Handwriting_Generation/models/conditional.pt")
    print("loaded")
    input_text = "an input string"
    #print(settings)
    plot_data = i_utils.sample_fixed_sequence(settings, model, truth_text=input_text)

    return plot_data


def recognize_stroke(stroke):
    # Input:
    #   stroke - numpy 2D-array (T x 3)

    # Output:
    #   text - str
    return 'welcome to lyrebird'


#stroke = generate_unconditionally()
#v_utils.plot_stroke(stroke)

stroke1 =generate_conditionally()
v_utils.plot_stroke(stroke1)
