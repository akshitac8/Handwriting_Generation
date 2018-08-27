import torch
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
# Named tuple to hold the data
#DataContainer = namedtuple("DataContainer", ["strokes", "texts", "onehots"])


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

    return conf, texts, texts_one_hot


def load_data(conf, data_path="data", validate=False):
    # Load the array of strokes
    raw_strokes = np.load('/home/aditya/version-control/workspace/Handwriting-Generation-Using-Recurrent-Neural-Networks/data/strokes.npy' , encoding="latin1")
    # Load the list of sentences
    with open('/home/aditya/version-control/workspace/Handwriting-Generation-Using-Recurrent-Neural-Networks/data/sentences.txt' ) as f:
        raw_texts = f.readlines()

    # We will compute the mean ratio len_stroke / len_onehot
    stroke_counter, text_counter = 0, 0
    # We remove pairs of (stroke, text) where len(stroke) < conf.bptt
    strokes, texts = [], []
    for s, t in zip(raw_strokes, raw_texts):
        # Put strokes in a list, throw out those with length smaller than bptt + 1
        # recall bptt is the seq len through which we backpropagate
        # + 1 comes from the tagret which is offset by +1
        if s.shape[0] > conf.bptt + 1:
            strokes.append(s)
            texts.append(t)
            # Update our stroke and text counters
            stroke_counter += s.shape[0]
            text_counter += len(t)

    # Compute the mean ratio len_stroke / len_onehot (used in conditional generation)
    conf.stroke_onehot_ratio = int(stroke_counter / text_counter)

    # Further processing of the text data in conditional mode (character removing, onehot encoding)
    #if validate:
    conf, texts, onehots = process_text_data(conf, texts)

    # Shuffle for good measure
    rng_state = np.random.get_state()
    np.random.shuffle(strokes)
    #print(strokes.type())  
    np.random.set_state(rng_state)
    np.random.shuffle(texts)
    np.random.set_state(rng_state)
    np.random.shuffle(onehots)

        # No train/val split as the losses are not very indicative of quality
        # and we prefer validating on qualitative visual inspection
     #   data_container = DataContainer(strokes=strokes, texts=texts, onehots=onehots)

    return strokes, texts, onehots


def get_random_conditional_training_batch(conf, strokes, texts, onehots):
    
    #strokes = data.strokes
    #onehots = data.onehots

    assert len(strokes) == len(onehots)

    strokes_dim = strokes[0].shape[-1]
    onehot_dim = onehots[0].shape[-1]
    onehot_len = conf.bptt // conf.stroke_onehot_ratio

    # Initialize numpy arrays where we'll fill features and targets
    # We format data as batch first (cf. models.ConditionalHandwritingRNN for explanation)
    X_npy = np.zeros((conf.batch_size, conf.bptt, strokes_dim), dtype=np.float32)
    Y_npy = np.zeros((conf.batch_size, conf.bptt, strokes_dim), dtype=np.float32)
    onehot_npy = np.zeros((conf.batch_size, onehot_len, onehot_dim), dtype=np.float32)

    # Get the list of sequences corresponding to the batch
    idxs = np.random.randint(0, len(strokes), conf.batch_size)

    for batch_idx, idx in enumerate(idxs):
        stroke = strokes[idx]
        onehot = onehots[idx]

        # We only use the start of the stroke
        # Otherwise, the network would have to also learn where to
        # start the alignment
        X_npy[batch_idx, ...] = stroke[:conf.bptt]
        Y_npy[batch_idx, ...] = stroke[1: conf.bptt + 1]
        onehot_npy[batch_idx, :onehot.shape[0], :] = onehot[:onehot_len, :]

    X_tensor = torch.from_numpy(X_npy)
    Y_tensor = torch.from_numpy(Y_npy)
    onehot_tensor = torch.from_numpy(onehot_npy)

    # Wrap to Autograd Variable 
    #X_tensor = Variable(X_tensor)
    #Y_tensor = Variable(Y_tensor)
    #onehot_tensor = Variable(onehot_tensor)

    # Check tensor dimensions
    assert X_tensor.size(0) == conf.batch_size
    assert Y_tensor.size(0) == conf.batch_size
    assert onehot_tensor.size(0) == conf.batch_size

    assert X_tensor.size(1) == conf.bptt
    assert Y_tensor.size(1) == conf.bptt

    assert X_tensor.size(2) == 3
    assert Y_tensor.size(2) == 3

    return X_tensor, Y_tensor, onehot_tensor


def train_step(conf, model, X_var, Y_var, optimizer, onehot=None):
   
    model.train()

    # Reset gradients
    optimizer.zero_grad()

    # Initialize hidden
    print('X_var',X_var.size())
    hidden = model.initHidden(X_var.size(0))
    #print("hidden",hidden.size())
    # Forward pass
    #mdnparams, e_logit, _ = model(X_var, hidden, onehot=onehot)
    mu1, mu2, sigma1, sigma2, rho, pi_mixprob, e_logit, hidden = model(X_var, hidden, onehot=onehot)

    # Flatten target
    target = Y_var.view(-1, 3).contiguous()
    # Extract eos, X1, X2
    eos, X1, X2 = target.split(1, dim=1)

    # Compute nll loss for next stroke 2D prediction
    nll = gaussian_2Dnll(X1, X2, mu1, mu2, sigma1, sigma2, rho, pi_mixprob)
    # Compute binary classification loss for end of sequence tag
    loss_bce = nn.BCEWithLogitsLoss(size_average=True)(eos, e_logit)

    # Sum the losses
    total_loss = (nll + loss_bce)

    d_loss = {"nll": nll.data.cpu().numpy(),
              "bce": loss_bce.data.cpu().numpy(),
              "total": total_loss.data.cpu().numpy()}

    # Backward pass
    total_loss.backward(retain_graph=True)
    # Gradient clipping
    torch.nn.utils.clip_grad_norm(model.parameters(), 10)
    optimizer.step()
    #hidden.detach()

    return d_loss

def logsumexp(x):

    assert x.dim() == 2
    x_max, x_max_idx = x.max(dim=-1, keepdim=True)
    logsum = x_max + torch.log((x - x_max).exp().sum(dim=-1, keepdim=True))
    return logsum


def compute_Z(X1, X2, mu1, mu2, log_sigma1, log_sigma2, rho):
    
    term1 = torch.pow((X1 - mu1) / log_sigma1.exp(), 2)
    term2 = torch.pow((X2 - mu2) / log_sigma2.exp(), 2)
    term3 = -2 * rho * (X1 - mu1) * (X2 - mu2) / (log_sigma1.exp() * log_sigma2.exp())
    Z = term1 + term2 + term3

    return Z


def gaussian_2Dnll(X1, X2, mu1, mu2, sigma1, sigma2, rho, pi_mixprob):
   
    X1 = X1.expand_as(mu1)
    X2 = X2.expand_as(mu1)

    Z = compute_Z(X1, X2, mu1, mu2, sigma1, sigma2, rho)

    # Rewrite likelihood part of Eq. 26 as logsumexp for stability
    pi_term = F.log_softmax(pi_mixprob)
    Z_term = -0.5 * Z / (1 - torch.pow(rho, 2))
    sigma_term = - torch.log(2 * float(np.pi) * sigma1.exp() * sigma2.exp() * torch.sqrt(1 - torch.pow(rho, 2)))

    exp_term = pi_term + Z_term + sigma_term
    nll = -logsumexp(exp_term).squeeze().mean()

    return nll



conf = configurations.get_args()

data0,data1,data3 = load_data(conf)

# Model specifications
input_size = data0[0].shape[1]
print ("Input Size : ",input_size)
onehot_dim = data3[0].shape[-1]
print ("Onehot dimensions : ",onehot_dim)
#output_size = 3 * conf.n_gaussian + 1
#print ("Output Size : ",output_size)
model = m.CondHandRNN(input_size, onehot_dim=onehot_dim)
optimizer = torch.optim.Adam(model.parameters(),lr=1E-3)

loss = ""
d_monitor = defaultdict(list)

# ***************** Training *************************
print("training")
for epoch in tqdm(range(conf.nb_epoch), desc="Training"):

    # Track the training losses over an epoch
    d_epoch_monitor = defaultdict(list)

    # Loop over batches
    desc = "Epoch: %s -- %s" % (epoch, loss)
    for batch in tqdm(range(conf.n_batch_per_epoch), desc=desc):

        X_var, Y_var, onehot_var = get_random_conditional_training_batch(conf, data0, data1, data3)
        #print (X_var.shape, " X_var")
        #print (Y_var.shape, " Y_var")
        #print (onehot_var, " onehot_var")

        # Train step.
        d_loss = train_step(conf, model, X_var, Y_var, optimizer, onehot=onehot_var)

        d_epoch_monitor["bce"].append(d_loss["bce"])
        d_epoch_monitor["nll"].append(d_loss["nll"])
        d_epoch_monitor["total"].append(d_loss["total"])

    # Update d_monitor with the mean over an epoch
    for key in d_epoch_monitor.keys():
        d_monitor[key].append(np.mean(d_epoch_monitor[key]))
    # Prepare loss to update progress bar
    loss = "Total : %.3g  " % (d_monitor["total"][-1])

    plot_data = i_utils.sample_fixed_sequence(conf, model)
    v_utils.plot_stroke(plot_data.stroke, "Plots/conditional_training/epoch_%s.png" % epoch)

    # Move model to cpu before training to allow inference on cpu
    if epoch % 5 == 0:


        # Move model to cpu before training to allow inference on cpu
        model.cpu()
        torch.save(model, conf.conditional_model_path)

print("Finished")
