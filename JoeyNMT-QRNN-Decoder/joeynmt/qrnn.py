import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from torch.nn import RNNBase
from torch.nn.utils.rnn import pad_packed_sequence
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import PackedSequence

from joeynmt.ifo_pooling import IFOPooling
from joeynmt.f_pooling import FPooling

import numbers


class QRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1,
                 batch_first=False, dropout=0., bidirectional=False,
                 pooling='ifo', kernel_size=2, first_kernel_size=None,
                 zoneout=0., dense=False, **kwargs):
        super(QRNN, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.dropout = dropout  # Dropout between layers
        self.bidirectional = bidirectional
        self.pooling = pooling
        self.kernel_size = kernel_size
        if first_kernel_size is None:
            self.first_kernel_size = kernel_size
        else:
            self.first_kernel_size = first_kernel_size
        self.zoneout = zoneout  # Zoneout (layer internal)
        self.dense = dense  # Concatenate input of layer to output
        
        num_directions = 2 if bidirectional else 1
        self.num_directions = num_directions
        
        invalid_dropout = not isinstance(dropout, numbers.Number) or \
            not 0 <= dropout <= 1 or isinstance(dropout, bool)
        
        invalid_zoneout = not isinstance(zoneout, numbers.Number) or \
            not 0 <= zoneout <= 1 or isinstance(zoneout, bool)
        
        if invalid_dropout:
            raise ValueError(
                "dropout should be a number in range [0, 1] representing the "
                "probability of an element being zeroed between layers"
                )
        if invalid_zoneout:
            raise ValueError(
                "zoneout should be a number in range [0, 1] representing the "
                "probability of an element being zeroed in the pooling layer"
                )

        self.layer_dropout = nn.Dropout(p=dropout)
        self.layers = nn.ModuleDict()
        
        current_input_size = input_size
        
        # The following code closely follows the PyTorch RNN implementation
        for layer in range(num_layers):
            if layer == 0:
                current_kernel_size = self.first_kernel_size
            else:
                current_kernel_size = self.kernel_size

            for direction in range(num_directions):
                reverse = True if direction == 1 else False
                suffix = '_reverse' if direction == 1 else ''
                module_name = 'layer_{}'.format(layer) + suffix
                self.layers[module_name] = \
                    QRNNLayer(current_input_size, hidden_size,
                              kernel_size=current_kernel_size,
                              zoneout=zoneout, reverse=reverse,
                              pooling=self.pooling)

            if dense:
                current_input_size += hidden_size*num_directions
            else:
                current_input_size = hidden_size*num_directions
    
    def __repr__(self):
        return ("QRNN(" +\
                "input_size={}, hidden_size={}, num_layers={}," +\
                " dropout={}, bidirectional={}, pooling={}," +\
                " kernel_size={}, first_kernel_size={}, zoneout={}, dense={}"
            ).format(self.input_size, self.hidden_size, self.num_layers,
                     self.dropout, self.bidirectional, self.pooling,
                     self.kernel_size, self.first_kernel_size, self.zoneout,
                     self.dense)


    def forward(self, input, hx=None):
        is_packed = isinstance(input, PackedSequence)
        
        if is_packed:
            input, lengths = \
                pad_packed_sequence(input, batch_first=self.batch_first)
        else:
            if self.batch_first:
                batch_size, timesteps, _ = input.size()
            else:
                timesteps, batch_size, _ = input.size()
            
            lengths = torch.LongTensor(batch_size)
            lengths.fill_(timesteps-1)  # Indexing starts with 0
        
        # Make input batch-first
        if not self.batch_first:
            input = input.permute(1, 2, 0)
        else:
            input = input.permute(0, 2, 1)
        # input: [batch, features, time]
        
        batch_size, features, timesteps = input.size()
        batch_idx = torch.arange(batch_size)
        last_hidden_states = [[] for _ in range(self.num_layers)]

        for layer in range(self.num_layers):
            # print(layer)
            directions = []
            for direction in range(self.num_directions):
                suffix = '_reverse' if direction == 1 else ''
                module_name = 'layer_{}'.format(layer) + suffix
                qrnn_layer = self.layers[module_name]
                
                input_prime = qrnn_layer(self.layer_dropout(input.contiguous()))
                directions.append(input_prime)
                
                t_last = 0 if direction == 1 else lengths-1
                last_hidden_state = input_prime[batch_idx, :, t_last]
                last_hidden_states[layer].append(last_hidden_state.unsqueeze(2))
            
            last_hidden_states[layer] = torch.cat(last_hidden_states[layer], dim=1)
            # Dense connections
            input_prime = torch.cat(directions, dim=1)
            if self.dense and not layer == len(self.layers)-1:
                input = torch.cat([input, input_prime], dim=1)
            else:
                input = input_prime
        
        # Restore original dims
        if not self.batch_first:
            input = input.permute(2, 0, 1)
        else:
            input = input.permute(0, 2, 1)
        
        if is_packed:
            output = \
                pack_padded_sequence(input, lengths, batch_first=self.batch_first)
        else:
            output = input
        
        # Layers*directions x batch x hidden
        h_n = torch.cat(last_hidden_states, dim=2).permute(2, 0, 1)
        
        return output, h_n


class QRNNLayer(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size=2, zoneout=0.0,
                 reverse=False, pooling='ifo', **kwargs):
        super(QRNNLayer, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.zoneout_p = zoneout
        self.reverse = reverse
        self.pooling = pooling

        self.z_conv = nn.Conv1d(input_size, hidden_size, kernel_size)
        self.f_conv = nn.Conv1d(input_size, hidden_size, kernel_size)
        self.o_conv = nn.Conv1d(input_size, hidden_size, kernel_size)
        self.i_conv = nn.Conv1d(input_size, hidden_size, kernel_size)
        self.zoneout = nn.Dropout(p=zoneout)
        
        if pooling == "f":
            self.f = FPooling()
            self.pooling = self.f_pooling
        elif pooling == "fo":
            self.f = FPooling()
            self.pooling = self.fo_pooling
        elif pooling == "ifo":
            self.ifo = IFOPooling()
            self.pooling = self.ifo_pooling
        else:
            raise ValueError("Invalid pooling: {}".format(pooling))


    def forward(self, input):
        # Assumes inputs of shape [minibatch, features, timesteps]
        training = self.training
        self.training = False  # Switch off dropout rescaling
        if self.reverse:
            input = input.flip(2)
        
        padding = torch.zeros(
            input.shape[0], input.shape[1], self.kernel_size-1,
            dtype=input.dtype, device=input.device
            )
        
        input = torch.cat([padding, input], dim=2).contiguous()
        z = torch.tanh(self.z_conv(input))
        f = 1 - self.zoneout(1 - torch.sigmoid(self.f_conv(input)))
        pooled = self.pooling(input, z, f)
        
        self.training = training
        return pooled


    def f_pooling(self, x, z, f):
        #h = [(1-f[:, :, 0]) * z[:, :, 0]]

        #for t in range(1, z.shape[2]):
            #h.append(f[:, :, t] * h[t-1] + (1-f[:, :, t]) * z[:, :, t])

        #return torch.cat([h_t.unsqueeze(2) for h_t in h], dim=2).contiguous()
        return self.f(f, z)


    def fo_pooling(self, x, z, f):
        o = torch.sigmoid(self.o_conv(x))

        #c = [(1-f[:, :, 0]) * z[:, :, 0]]

        #for t in range(1, z.shape[2]):
            #c.append(f[:, :, t] * c[t-1] + (1-f[:, :, t]) * z[:, :, t])

        #return o*torch.cat([c_t.unsqueeze(2) for c_t in c], dim=2).contiguous()
        c = self.f(f, z)
        return o*c


    def ifo_pooling(self, x, z, f):
        o = torch.sigmoid(self.o_conv(x))
        i = torch.sigmoid(self.i_conv(x))
        
        c = self.ifo(f, z, i)
        return o*c

        #c = [i[:, :, 0] * z[:, :, 0]]

        #for t in range(1, z.shape[2]):
            #c.append(f[:, :, t] * c[t-1] + i[:, :, t] * z[:, :, t])

        #return o*torch.cat([c_t.unsqueeze(2) for c_t in c], dim=2).contiguous()
