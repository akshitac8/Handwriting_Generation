import numpy as np
import torch
from torch.autograd import Variable

def getuncond_sequence(conf, model):
    sample = [1., 0., 0.]  
    sample = torch.Tensor(sample).view(1,1,3)   

    sample = Variable(sample)
    hidden = model.initHidden(sample.size(1))

    list_stroke = []
    for i in range(conf.sampling_len):

        mu1, mu2, sigma1, sigma2, rho, pi_mixprob, e_logit, hidden = model(sample, hidden)

        x1, x2, eos, mu1, mu2, sigma1, sigma2, rho = stroke_sampling(conf, mu1, mu2, sigma1, sigma2, rho, pi_mixprob, e_logit)

        sample = torch.cat([eos, x1, x2], -1).view(1,1,3)
        list_stroke.append(sample.squeeze().data.cpu().numpy())

    arr_stroke = np.array(list_stroke)

    return arr_stroke


def sample_fixed_sequence(conf, model, truth_text="Welcome to lyrebird"):
    

    print(truth_text)
    truth_onehot = np.zeros((1, len(truth_text), len(conf.d_char_to_idx.keys())), dtype=np.float32)
    for idx, char in enumerate(truth_text):
        truth_onehot[0, idx, conf.d_char_to_idx[char]] = 1.0

    # Reconstruct text from onehot
    reconstructed_text = ""
    for i in range(truth_onehot.shape[1]):
        char_idx = np.argmax(truth_onehot[0, i, :])
        char = conf.d_idx_to_char[char_idx]
        reconstructed_text += char

    # Sanity check to make sure there is no mixup between onehot and text
    assert truth_text == reconstructed_text

    # Prepare truth_onehot for model
    onehot = torch.from_numpy(truth_onehot)

    # Prepare input sequence
    sample = [1., 0., 0.]  # eos = 1 to tell the model to generate a new sample dx and dy initialized at 0
    sample = torch.Tensor(sample).view(1,1,3)  # format in (seq_len, batch_size, n_features) mode

    # Prepare kappa
    running_kappa = torch.zeros(1, model.num_wgauss, 1)

    sample = Variable(sample)
    onehot = Variable(onehot)
    running_kappa = Variable(running_kappa)

    # Prepare hidden var
    hidden = model.initHidden(sample.size(0))

    list_stroke = []
    list_density = []
    list_phi = []
    list_window = []
    list_kappa = []
    for i in range(conf.sampling_len):

        # Set training flag to false to indicate we are doing inference.
        mu1, mu2, log_sigma1, log_sigma2, rho, pi_logit, e_logit, hidden = model(sample, hidden, onehot, training=False, running_kappa=running_kappa)

        # End condition when mean of densityian window is longer than the len of the one hot sequence
        mean_kappa = running_kappa.squeeze().mean().data.cpu().numpy()
        if mean_kappa + 1 > onehot.size(1):
            break

        # Sample a new data point
        x1, x2, eos, mu1, mu2, sigma1, sigma2, rho = stroke_sampling(conf, mu1, mu2, log_sigma1, log_sigma2, rho, pi_logit, e_logit)

    
        # Get the tensors related to the attention window for plotting
        window = model.window.data.cpu().numpy()
        phi = model.phi.data.cpu().numpy()
        kappa = model.new_kappa.data.cpu().numpy()

        # Updata sample and kappa
        sample = torch.cat((eos, x1, x2), -1).view(1,1,3)
        running_kappa = Variable(torch.from_numpy(kappa))

        # Store parameters for plot
        list_stroke.append(sample.squeeze().data.cpu().numpy())
        list_window.append(window)
        list_phi.append(phi)
        list_kappa.append(kappa[0].T)
        list_density.append(np.hstack([mu1, mu2, sigma1, sigma2, rho]))

    arr_stroke = np.stack(list_stroke, axis=0)
    arr_window = np.concatenate(list_window, axis=0)
    arr_phi = np.stack(list_phi, axis=0)
    arr_density = np.stack(list_density, axis=0)
    arr_kappa = np.concatenate(list_kappa, axis=0)


    return arr_stroke, arr_window, arr_phi, arr_density, arr_kappa, truth_text 


def stroke_sampling(conf, mu1, mu2, log_sigma1, log_sigma2, rho, pi_logit, e_logit=None):
    
    #print (pi_logit, " : pi_logit")
    pi_logit[torch.isnan(pi_logit)] = 0
    #print (pi_logit, " : pi_logit")
    a=torch.from_numpy(np.array(pi_logit.size(0) * (1.0 + conf.bias)))
    #print(a.type())
    g_m = torch.softmax(a, dim=0)
    # To sample from mixture of gaussian, sample a component with bernoulli and weight pi
    #print("Done")
    idx = torch.multinomial(g_m, 1)
    #print (idx , " : idx")
    idx=idx.view(-1,1)
    #print(mu1.size())
    # Select the gaussian parameters corresponding to idxs
    mu1 = mu1.gather(1, idx)
    mu2 = mu2.gather(1, idx)
    sigma1 = (log_sigma1.gather(1, idx) - conf.bias).exp()
    sigma2 = (log_sigma2.gather(1, idx) - conf.bias).exp()
    rho = rho.gather(1, idx)

    eps1 = torch.normal(mean=0., std=torch.ones(1)).view(1, -1)
    eps2 = torch.normal(mean=0., std=torch.ones(1)).view(1, -1)

    eps1 = Variable(eps1)
    eps2 = Variable(eps2)

    x1 = sigma1 * eps1 + mu1
    x2 = sigma2 * (rho * eps1 + torch.sqrt(1 - rho * rho) * eps2) + mu2

    # Move MDN params to cpu for use in plotting
    mu1 = mu1.data.cpu().numpy()
    mu2 = mu2.data.cpu().numpy()
    sigma1 = sigma1.data.cpu().numpy()
    sigma2 = sigma2.data.cpu().numpy()
    rho = rho.data.cpu().numpy()

    if e_logit is None:

        return x1, x2, mu1, mu2, sigma1, sigma2, rho

    else:
        #print(e_logit.type())
        #e_logit = e_logit.var.detach().numpy()
        s_m=torch.sigmoid(e_logit)
        #print(s_m)
        s_m[torch.isnan(s_m)] = 0
        eos = torch.bernoulli(s_m)
        #print(eos,"eos")
        return x1, x2, eos, mu1, mu2, sigma1, sigma2, rho
