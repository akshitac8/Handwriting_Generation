from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

#################### UnConditional Handwriting generation #####################
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
        rho= torch.tanh(rho)
        return mu1, mu2, sigma1, sigma2, rho, pi_mixprob, eos, hidden

    def initHidden(self, batch_size):

       return torch.zeros(self.num_layers, batch_size, self.hidden_dim)

  
######################## Conditional Handwriting generation #############            

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
                    nn.init.normal_(p, mean=0, std=0.075)
        for p_name, p in self.window_layer.named_parameters():
            if "bias" in p_name:
                # Custom initi for bias so that the kappas do not grow too fast
                # and prevent sequence alignment
                nn.init.normal_(p, mean=-4.0, std=0.1)

    def forward(self, x_input, hidden, onehot, training=True, running_kappa=None):
        #print(x_input.size())
        #print(len(hidden))
        #print(onehot.size())
        #print(hidden[0].size())
        #print(hidden[1].size())
        # Initialize U to compute the gaussian windows
        U = Variable(torch.arange(0, onehot.size(1)), requires_grad=False).float()
        U = U.view(1, 1, 1, -1)  # prepare for broadcasting
          
        # Pass input to first layer
        out0, hidden[0] = self.rnn_layer0(x_input, hidden[0])
        # Compute the gaussian window parameters
        alpha, beta, kappa = torch.exp(self.window_layer(out0)).unsqueeze(-1).split(self.num_wgauss, -2)
        #print (len(alpha), " alpha")
        #print (len(beta), " beta")
        #print (len(kappa), " kappa")
        
        # In training mode compute running_kappa = cumulative sum of kappa
        if training:
            running_kappa = kappa.cumsum(1)
        # Otherwise, update the previous kappa
        else:
            assert running_kappa is not None
            running_kappa = running_kappa.unsqueeze(1) + kappa
        # Compute the window
        #print(alpha.type())
        #print(running_kappa.type())
        #print(beta.type())
        phi = alpha * torch.exp(-beta * (running_kappa - U).pow(2))
        phi = phi.sum(-2)
        #print (phi.shape, " phi")
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
                #print("hi in forwrd")
                out = torch.cat([out, window, x_input], -1)

        # Flatten model output so that the same operation is applied to each time step
        out = out.contiguous().view(-1, out.size(-1))
        out = self.mdn_layer(out)

        mu1, mu2, log_sigma1, log_sigma2, rho,pi_logit, e_logit = torch.split(out, self.num_gauss, dim=1)
        mu1, mu2, sigma1, sigma2, rho, pi_mixprob, eos= out.split(self.num_gauss, dim=1)
        rho= torch.tanh(rho)
        # Store Gaussian Mixture params in a namedtuple
       
        return mu1, mu2, log_sigma1, log_sigma2, rho, pi_logit, e_logit, hidden

        

    def initHidden(self, batch_size):

        return [(torch.zeros(1, batch_size, self.hidden_dim)), (torch.zeros(1, batch_size, self.hidden_dim)) ]

        
############ Text Recognition ###############################


class Encoder(nn.Module):
    def __init__(self, input_size, embedding_size,hidden_size, n_layers=1,bidirec=False):
        super(Encoder, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        
        self.embedding = nn.Embedding(input_size, embedding_size)
        
        if bidirec:
            self.n_direction = 2 
            self.gru = nn.GRU(embedding_size, hidden_size, n_layers, batch_first=True, bidirectional=True)
        else:
            self.n_direction = 1
            self.gru = nn.GRU(embedding_size, hidden_size, n_layers, batch_first=True)
    
    def init_hidden(self, inputs):
        hidden = Variable(torch.zeros(self.n_layers * self.n_direction, inputs.size(0), self.hidden_size))
        return hidden.cuda() if USE_CUDA else hidden
    
    def init_weight(self):
        self.embedding.weight = nn.init.xavier_uniform_(self.embedding.weight)
        self.gru.weight_hh_l0 = nn.init.xavier_uniform_(self.gru.weight_hh_l0)
        self.gru.weight_ih_l0 = nn.init.xavier_uniform_(self.gru.weight_ih_l0)
    
    def forward(self, inputs, input_lengths):
        """
        inputs : B, T (LongTensor)
        input_lengths : real lengths of input batch (list)
        """
        hidden = self.init_hidden(inputs)
        
        embedded = self.embedding(inputs)
        packed = pack_padded_sequence(embedded, input_lengths, batch_first=True)
        outputs, hidden = self.gru(packed, hidden)
        outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True) # unpack (back to padded)
                
        if self.n_layers > 1:
            if self.n_direction == 2:
                hidden = hidden[-2:]
            else:
                hidden = hidden[-1]
        
        return outputs, torch.cat([h for h in hidden], 1).unsqueeze(1)
    
    
    
class Decoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, n_layers=1, dropout_p=0.1):
        super(Decoder, self).__init__()
        
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        
        # Define the layers
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.dropout = nn.Dropout(dropout_p)
        
        self.gru = nn.GRU(embedding_size + hidden_size, hidden_size, n_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size * 2, input_size)
        self.attn = nn.Linear(self.hidden_size, self.hidden_size) # Attention
    
    def init_hidden(self,inputs):
        hidden = Variable(torch.zeros(self.n_layers, inputs.size(0), self.hidden_size))
        return hidden.cuda() if USE_CUDA else hidden
    
    
    def init_weight(self):
        self.embedding.weight = nn.init.xavier_uniform_(self.embedding.weight)
        self.gru.weight_hh_l0 = nn.init.xavier_uniform_(self.gru.weight_hh_l0)
        self.gru.weight_ih_l0 = nn.init.xavier_uniform_(self.gru.weight_ih_l0)
        self.linear.weight = nn.init.xavier_uniform_(self.linear.weight)
        self.attn.weight = nn.init.xavier_uniform_(self.attn.weight)
#         self.attn.bias.data.fill_(0)
    
    def Attention(self, hidden, encoder_outputs, encoder_maskings):
        """
        hidden : 1,B,D
        encoder_outputs : B,T,D
        encoder_maskings : B,T # ByteTensor
        """
        hidden = hidden[0].unsqueeze(2)  # (1,B,D) -> (B,D,1)
        
        batch_size = encoder_outputs.size(0) # B
        max_len = encoder_outputs.size(1) # T
        energies = self.attn(encoder_outputs.contiguous().view(batch_size * max_len, -1)) # B*T,D -> B*T,D
        energies = energies.view(batch_size,max_len, -1) # B,T,D
        attn_energies = energies.bmm(hidden).squeeze(2) # B,T,D * B,D,1 --> B,T
        
#         if isinstance(encoder_maskings,torch.autograd.variable.Variable):
#             attn_energies = attn_energies.masked_fill(encoder_maskings,float('-inf'))#-1e12) # PAD masking
        
        alpha = F.softmax(attn_energies,1) # B,T
        alpha = alpha.unsqueeze(1) # B,1,T
        context = alpha.bmm(encoder_outputs) # B,1,T * B,T,D => B,1,D
        
        return context, alpha
    
    
    def forward(self, inputs, context, max_length, encoder_outputs, encoder_maskings=None, is_training=False):
        """
        inputs : B,1 (LongTensor, START SYMBOL)
        context : B,1,D (FloatTensor, Last encoder hidden state)
        max_length : int, max length to decode # for batch
        encoder_outputs : B,T,D
        encoder_maskings : B,T # ByteTensor
        is_training : bool, this is because adapt dropout only training step.
        """
        # Get the embedding of the current input word
        embedded = self.embedding(inputs)
        hidden = self.init_hidden(inputs)
        if is_training:
            embedded = self.dropout(embedded)
        
        decode = []
        # Apply GRU to the output so far
        for i in range(max_length):

            _, hidden = self.gru(torch.cat((embedded, context), 2), hidden) # h_t = f(h_{t-1},y_{t-1},c)
            concated = torch.cat((hidden, context.transpose(0, 1)), 2) # y_t = g(h_t,y_{t-1},c)
            score = self.linear(concated.squeeze(0))
            softmaxed = F.log_softmax(score,1)
            decode.append(softmaxed)
            decoded = softmaxed.max(1)[1]
            embedded = self.embedding(decoded).unsqueeze(1) # y_{t-1}
            if is_training:
                embedded = self.dropout(embedded)
            
            # compute next context vector using attention
            context, alpha = self.Attention(hidden, encoder_outputs, encoder_maskings)
            
        #  column-wise concat, reshape!!
        scores = torch.cat(decode, 1)
        return scores.view(inputs.size(0) * max_length, -1)
    
    def decode(self, context, encoder_outputs):
        start_decode = Variable(LongTensor([[target2index['<s>']] * 1])).transpose(0, 1)
        embedded = self.embedding(start_decode)
        hidden = self.init_hidden(start_decode)
        
        decodes = []
        attentions = []
        decoded = embedded
        while decoded.data.tolist()[0] != target2index['</s>']: # until </s>
            _, hidden = self.gru(torch.cat((embedded, context), 2), hidden) # h_t = f(h_{t-1},y_{t-1},c)
            concated = torch.cat((hidden, context.transpose(0, 1)), 2) # y_t = g(h_t,y_{t-1},c)
            score = self.linear(concated.squeeze(0))
            softmaxed = F.log_softmax(score,1)
            decodes.append(softmaxed)
            decoded = softmaxed.max(1)[1]
            embedded = self.embedding(decoded).unsqueeze(1) # y_{t-1}
            context, alpha = self.Attention(hidden, encoder_outputs,None)
            attentions.append(alpha.squeeze(1))
        
        return torch.cat(decodes).max(1)[1], torch.cat(attentions)
