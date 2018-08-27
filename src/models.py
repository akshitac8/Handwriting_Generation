from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class UncondHandRNN(nn.Module):
    
    def __init__(self, input_dim, hidden_dim = 512,num_gauss=20, num_layers =3):
        super(UncondHandRNN, self).__init__()   
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_gauss = num_gauss
        self.output_dim = 1 + 6*self.num_gauss
        self.num_gauss = num_gauss

        self.gru = nn.GRU(input_dim,
                             self.hidden_dim,
                             num_layers=num_layers,
                             dropout=0.02,
                             batch_first=True)
        self.mdn_linear = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, input, hidden=None):
        #print("input",input.size())

        #input = input.view(-1,1,3)
        out, hidden = self.gru(input, hidden)
        out = out.contiguous().view(-1, out.size(-1))
        out = self.mdn_linear(out)
        #print('out',out.size())
        mu1, mu2, sigma1, sigma2, rho, pi_mixprob, eos= out.split(self.num_gauss, dim=1)
        rho= nn.functional.tanh(rho)
        return mu1, mu2, sigma1, sigma2, rho, pi_mixprob, eos, hidden

    def initHidden(self, batch_size):

       return torch.zeros(self.num_layers, batch_size, self.hidden_dim)

            


class CondHandRNN(nn.Module):
    

    def __init__(self, input_dim,onehot_dim,hidden_dim = 256, num_gauss=20, num_layers = 2, num_wgauss=10):
        super(CondHandRNN, self).__init__()

        # Params
        self.input_dim = input_dim        
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.onehot_dim = onehot_dim
        self.num_gauss = num_gauss
        self.num_wgauss = num_wgauss
        self.output_dim = 1 + 3*self.num_gauss

        

        #########################
        # Layers definition
        #########################

        model0_input_dim = self.input_dim
        modelx_input_dim = self.hidden_dim + self.input_dim + self.onehot_dim

        # 1) Use batch first because we use torch.bmm afterwards. Without batch first
        # we would have to transpose the tensor which may lead to confusion
        # 2) Explicitly split into num_layers layers so that we can use skip connections
        self.rnn_layer0 = nn.GRU(model0_input_dim, self.hidden_dim, batch_first=True)

        for k in range(1, self.num_layers):
            setattr(self, "rnn_layer%s" % k, nn.GRU(modelx_input_dim, self.hidden_dim, batch_first=True))

        self.window_layer = nn.Linear(hidden_dim, 3 * self.num_wgauss)
        self.mdn_layer = nn.Linear(hidden_dim, 1 + 6 * self.num_gauss)

        list_layers = [getattr(self, "rnn_layer%s" % k) for k in range(self.num_layers)]
        list_layers += [self.window_layer, self.mdn_layer]

        #########################
        # Custom initialization
        #########################
        for layer in list_layers:
            for p_name, p in layer.named_parameters():
                if "weight" in p_name:
                    # Graves-like initialization
                    # (w/o the truncation which does not have much influence on the results)
                    nn.init.normal(p, mean=0, std=0.075)
        for p_name, p in self.window_layer.named_parameters():
            if "bias" in p_name:
                # Custom initi for bias so that the kappas do not grow too fast
                # and prevent sequence alignment
                nn.init.normal(p, mean=-4.0, std=0.1)

    def forward(self, x_input, hidden, onehot, training=True, running_kappa=None):
        print(x_input.size())
        print(len(hidden))
        print(onehot.size())
        print(hidden[0].size())
        print(hidden[1].size())
        # Initialize U to compute the gaussian windows
        U = Variable(torch.arange(0, onehot.size(1)), requires_grad=False)
        U = U.view(1, 1, 1, -1)  # prepare for broadcasting
          
        # Pass input to first layer
        out0, hidden[0] = self.rnn_layer0(x_input, hidden[0])
        # Compute the gaussian window parameters
        alpha, beta, kappa = torch.exp(self.window_layer(out0)).unsqueeze(-1).split(self.num_wgauss, -2)
        print (len(alpha), " alpha")
        print (len(beta), " beta")
        print (len(kappa), " kappa")
        
        # In training mode compute running_kappa = cumulative sum of kappa
        if training:
            running_kappa = kappa.cumsum(1)
        # Otherwise, update the previous kappa
        else:
            assert running_kappa is not None
            running_kappa = running_kappa.unsqueeze(1) + kappa
        print (len(kappa), " kappa")
        # Compute the window
        phi = alpha * torch.exp(-beta * (running_kappa - U).pow(2))
        phi = phi.sum(-2)
        print (phi.shape, " phi")
        window = torch.matmul(phi, onehot)

        # Save the last window/phi/kappa for plotting
        self.window = window[:, -1, :]
        self.phi = phi[:, -1, :]
        self.new_kappa = running_kappa[:, -1, :, :]

        # Next model layers
        out = torch.cat([out0, window, x_input], -1)
        for i in range(1, self.num_layers):
            out, hidden[i] = self.rnn_layer1(out, hidden[i])
            if i != self.num_layers - 1:
                print("hi in forwrd")
                out = torch.cat([out, window, x_input], -1)

        # Flatten model output so that the same operation is applied to each time step
        out = out.contiguous().view(-1, out.size(-1))
        out = self.mdn_layer(out)

        pi_logit, mu1, mu2, log_sigma1, log_sigma2, rho, e_logit = torch.split(out, self.num_gauss, dim=1)
        rho=F.tanh(rho)
        # Store Gaussian Mixture params in a namedtuple
       
        return pi_logit, mu1, mu2, log_sigma1, log_sigma2, rho, e_logit, hidden

        

    def initHidden(self, batch_size):

        return [(torch.zeros(1, batch_size, self.hidden_dim)), (torch.zeros(1, batch_size, self.hidden_dim)) ]

        