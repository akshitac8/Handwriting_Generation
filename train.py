import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import handwriting.models as m
import torch.nn.functional as F
import handwriting.inference_utils as i_utils
from utils import visualization_utils as v_utils
import configurations
print('Imported')

def load_data(conf, data_path="data", validate=False):

    raw_strokes = np.load('/home/aditya/version-control/workspace/Handwriting-Generation-Using-Recurrent-Neural-Networks/data/strokes.npy' , encoding="latin1")
    with open('/home/aditya/version-control/workspace/Handwriting-Generation-Using-Recurrent-Neural-Networks/data/sentences.txt' ) as f:
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

def get_random_unconditional_training_batch(conf, data):
    print("YEs i am in get_random_unconditional_training_batch ")
    strokes = data
    stroke_dim = data[0].shape[-1]
    #print('stroke_dim',stroke_dim)

    X_npy = np.zeros((conf.batch_size, conf.bptt, stroke_dim), dtype=np.float32)
    Y_npy = np.zeros((conf.batch_size, conf.bptt, stroke_dim), dtype=np.float32)

    idxs = np.random.randint(0, len(strokes), conf.batch_size)

    for batch_idx, idx in enumerate(idxs):

        stroke = strokes[idx]
        start = np.random.randint(0, stroke.shape[0] - conf.bptt - 1)
        X_npy[batch_idx, ...] = stroke[start: start + conf.bptt, :]
        Y_npy[batch_idx, ...] = stroke[start + 1: start + 1 + conf.bptt, :]

    X_tensor = torch.from_numpy(X_npy)
    Y_tensor = torch.from_numpy(Y_npy)

    #print(X_tensor.size())
    #print(Y_tensor.size())
    assert X_tensor.size(0) == conf.batch_size
    assert Y_tensor.size(0) == conf.batch_size

    assert X_tensor.size(1) == conf.bptt
    assert Y_tensor.size(1) == conf.bptt

    assert X_tensor.size(2) == 3
    assert Y_tensor.size(2) == 3

    return X_tensor, Y_tensor

def train_step(conf, model, X_var, Y_var, optimizer, onehot=None):
   
    model.train()
    optimizer.zero_grad()

    #print('X_var',X_var.size())
    hidden = model.initHidden(X_var.size(0))
    #print("hidden",hidden.size())
    
    mu1, mu2, sigma1, sigma2, rho, pi_mixprob, e_logit, hidden = model(X_var, hidden)

   
    target = Y_var.view(-1, 3).contiguous()
    eos, X1, X2 = target.split(1, dim=1)
    nll = gaussian_2Dnll(X1, X2, mu1, mu2, sigma1, sigma2, rho, pi_mixprob)
    
    loss_bce = nn.BCEWithLogitsLoss(size_average=True)(e_logit,eos)

   
    total_loss = (nll + loss_bce)

    d_loss = {"nll": nll.data.cpu().numpy(),
              "bce": loss_bce.data.cpu().numpy(),
              "total": total_loss.data.cpu().numpy()}

    total_loss.backward(retain_graph=True)
    torch.nn.utils.clip_grad_norm(model.parameters(), 5)
    optimizer.step()
    hidden.detach()

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
data = load_data(conf)

#print(data.shape[0])

# Model specifications
input_dim = data[0].shape[-1]
model = m.UncondHandRNN(input_dim)
print(model)
optimizer = torch.optim.Adam(model.parameters(),lr=1E-3)

loss = ""
d_monitor = defaultdict(list)

for epoch in tqdm(range(conf.nb_epoch), desc="Training"):

    # Track the training losses over an epoch
    d_epoch_monitor = defaultdict(list)

    # Loop over batches
    desc = "Epoch: %s -- %s" % (epoch, loss)
    for batch in tqdm(range(conf.n_batch_per_epoch), desc=desc):

        # Sample a batch (X, Y)
        print("Batch",batch)
        X_var, Y_var = get_random_unconditional_training_batch(conf, data)

        # Train step = forward + backward + weight update
        d_loss = train_step(conf, model, X_var, Y_var, optimizer)


        d_epoch_monitor["bce"].append(d_loss["bce"])
        d_epoch_monitor["nll"].append(d_loss["nll"])
        d_epoch_monitor["total"].append(d_loss["total"])

    # Sample a sequence to follow progress and save the plot
    #plot_data = i_utils.sample_unconditional_sequence(conf, model)
    #v_utils.plot_stroke(plot_data.stroke, "Plots/unconditional_training/epoch_%s.png" % epoch)

    # Update d_monitor with the mean over an epoch
    #for key in d_epoch_monitor.keys():
    #    d_monitor[key].append(np.mean(d_epoch_monitor[key]))
    # Prepare loss to update progress bar
    loss = "Total : %.3g " % (d_monitor["total"][-1])

    # Save the model at regular intervals
    if epoch % 5 == 0:
        model.cpu()
        torch.save(model, conf.unconditional_model_path)


print("Finished")


