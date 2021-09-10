# Author: Leander Girrbach
# Custom QRNN-Decoder implementation

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from torch.nn import RNNBase
from torch.nn.utils.rnn import pad_packed_sequence
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import PackedSequence

from joeynmt.qrnn import QRNNLayer
from joeynmt.decoders import RecurrentDecoder
import numbers


class QRNNDecoder(RecurrentDecoder, nn.Module):
    def __init__(self, emb_size=0, hidden_size=0, encoder=None,
                 vocab_size=0, emb_dropout=0., hidden_dropout=0.,
                 freeze=False, num_layers=1, batch_first=False,
                 dropout=0., pooling='ifo', kernel_size=2,
                 first_kernel_size=None, zoneout=0.,
                 dense=False, **kwargs):
        nn.Module.__init__(self)
        
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.dropout = dropout  # Dropout between layers
        self.pooling = pooling
        self.kernel_size = kernel_size
        self.encoder = encoder
        if first_kernel_size is None:
            self.first_kernel_size = kernel_size
        else:
            self.first_kernel_size = first_kernel_size
        self.zoneout = zoneout  # Zoneout (layer internal)
        self.dense = dense  # Concatenate input of layer to output
        
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
        self.layers = nn.ModuleList()
        
        current_input_size = self.emb_size
        self._output_size = vocab_size
        
        for layer in range(num_layers-1):
            if layer == 0:
                current_kernel_size = self.first_kernel_size
            else:
                current_kernel_size = self.kernel_size

            self.layers.append(
                QRNNDecoderLayer(
                    current_input_size, hidden_size,
                    kernel_size=current_kernel_size,
                    zoneout=zoneout,
                    pooling=self.pooling,
                    encoder_hidden_size=self.encoder.output_size
                    )
                )

            if dense:
                current_input_size += hidden_size
            else:
                current_input_size = hidden_size
        
        self.layers.append(
            AttentionLayer(
                current_input_size, hidden_size,
                kernel_size=current_kernel_size,
                zoneout=zoneout,
                hidden_dropout=hidden_dropout,
                pooling=self.pooling,
                encoder_hidden_size=self.encoder.output_size
                )
            )
        
        self.output_layer = nn.Linear(self.hidden_size, vocab_size, bias=False)


    def __repr__(self):
        return "QRNNDecoder(layers={}, hidden_size={})"\
            .format(self.num_layers, self.hidden_size)
    
    
    def forward(self, trg_embed, encoder_output, encoder_hidden, src_mask,
                unroll_steps, hidden= None, prev_att_vector= None, **kwargs):
        # initialize decoder hidden state from final encoder hidden state
        if hidden is None:
            self._init_hidden(encoder_hidden)
        else:
            ((hidden, context), input) = hidden
            self.hidden = hidden
            self.context = context
            self.input = input
        
        #if hidden is None:
            #hidden = self._init_hidden(encoder_hidden)

        # here we store all intermediate attention vectors (used for prediction)
        att_vectors = []
        att_probs = []

        batch_size = encoder_output.size(0)

        if prev_att_vector is None:
            with torch.no_grad():
                prev_att_vector = encoder_output.new_zeros(
                    [batch_size, 1, self.hidden_size])

        # unroll the decoder RNN for `unroll_steps` steps
        for i in range(unroll_steps):
            prev_embed = trg_embed[:, i].unsqueeze(2)  # batch, 1, emb
            prev_att_vector, att_prob = self._forward_step(
                prev_embed=prev_embed,
                # hidden = hidden,
                encoder_output=encoder_output,
                encoder_hidden=encoder_hidden,
                src_mask=src_mask)
            att_vectors.append(prev_att_vector)
            att_probs.append(att_prob)

        att_vectors = torch.cat(att_vectors, dim=1)
        # att_vectors: batch, unroll_steps, hidden_size
        att_probs = torch.cat(att_probs, dim=1)
        # att_probs: batch, unroll_steps, src_length
        last_hidden = self.hidden[-1][:, :, -unroll_steps:]
        # last_hidden = hidden[0][0][-1, :, :, -unroll_steps:]
        outputs = self.output_layer(last_hidden.transpose(1, 2))
        # outputs: batch, unroll_steps, vocab_size
        # print(outputs.shape)
        # return outputs, ((self.hidden, self.context), self.input), att_probs, att_vectors
        # hidden = ((self.hidden, self.context), self.input)
        # return outputs, hidden, att_probs, att_vectors
        return outputs, ((self.hidden, self.context), self.input), att_probs, att_vectors


    def _forward_step(self, prev_embed, encoder_output,
                      encoder_hidden, src_mask):
        # Input: [batch, features, t]
        # prev_embed: [batch, t, features]
        # (hidden, context), input = hidden
        #self.input = torch.cat([self.input, prev_embed], dim=2)
        # input = torch.cat([input, prev_embed], dim=2)
        #current_input = self.input
        # current_input = input
        
        #context = list(torch.split(context, 1, dim=0))
        #hidden = list(torch.split(hidden, 1, dim=0))
        self.input = torch.cat([self.input, prev_embed], dim=2)
        current_input = self.input
        
        for layer in range(self.num_layers-1):
            qrnn_layer = self.layers[layer]
            last_encoder_state = encoder_hidden[layer]
            
            h, c = self.hidden[layer], self.context[layer]
            # h, c = hidden[layer], context[layer]
            # h_new, c_new = qrnn_layer(self.layer_dropout(current_input),
                                      # h[0, :, :, -1:], c[0, :, :, -1:],
            h_new, c_new = qrnn_layer(self.layer_dropout(current_input),
                                      h[:, :, -1:], c[:, :, -1:],
                                      last_encoder_state)

            self.hidden[layer] = torch.cat([h, h_new], dim=2)
            self.context[layer] = torch.cat([c, c_new], dim=2)
            #hidden[layer] = torch.cat([h, h_new.unsqueeze(0)], dim=3)
            #context[layer] = torch.cat([c, c_new.unsqueeze(0)], dim=3)
            
            if self.dense:  # Dense convolutions not supported in the decoder
                raise NotImplementedError("Dense convolutions not yet supported")
            else:
                current_input = self.hidden[layer]

        # compute context vector using attention mechanism
        attention = self.layers[-1]
        h, c = self.hidden[-1], self.context[-1]
        # h, c = hidden[-1], context[-1]
        att_vector, (h_new, c_new), att_probs = attention(
            self.layer_dropout(current_input),
            # h[0, :, :, -1:], c[0, :, :, -1:],
            h[:, :, -1:], c[:, :, -1:],
            encoder_hidden[-1],
            encoder_output,
            src_mask
            )
        
        self.hidden[-1] = torch.cat([h, h_new], dim=2)
        self.context[-1] = torch.cat([c, c_new], dim=2)
        #hidden[-1] = torch.cat([h, h_new.unsqueeze(0)], dim=3)
        #context[-1] = torch.cat([c, c_new.unsqueeze(0)], dim=3)
        
        #hidden = torch.cat(hidden, dim=0)
        #context = torch.cat(context, dim=0)

        # return att_vector, ((hidden, context), input), att_probs
        return att_vector, att_probs
    
    
    def _init_hidden(self, encoder_final):
        with torch.no_grad():
            batch_size = encoder_final.size(1)
            device = encoder_final.device
            self.hidden = []
            self.context = []
            for _ in range(self.num_layers):
                h = torch.zeros(batch_size, self.hidden_size, 1,
                                device=device)
                c = torch.zeros(batch_size, self.hidden_size, 1,
                                device=device)
                self.hidden.append(h)
                self.context.append(c)
            
            #hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size, 1, device=device)
            #context = torch.zeros(self.num_layers, batch_size, self.hidden_size, 1, device=device)
        
            self.input = torch.zeros(batch_size, self.emb_size, 0, device=device)
            # input = torch.zeros(batch_size, self.emb_size, 0, device=device)
        
        return (self.hidden, self.context), self.input
        # return (hidden, context), input


class QRNNDecoderLayer(QRNNLayer):
    def __init__(self, *args, **kwargs):
        super(QRNNDecoderLayer, self).__init__(*args, **kwargs)
        encoder_hidden_size = kwargs['encoder_hidden_size']
        # For including the encoder hidden layer last states
        self.enc_proj_z = nn.Linear(encoder_hidden_size, self.hidden_size)
        self.enc_proj_f = nn.Linear(encoder_hidden_size, self.hidden_size)
        self.enc_proj_o = nn.Linear(encoder_hidden_size, self.hidden_size)
        self.enc_proj_i = nn.Linear(encoder_hidden_size, self.hidden_size)
        
        # An abbreviation
        self.s = torch.sigmoid


    def forward(self, input, hidden, context, last_encoder_state):
        input, z, f = self.convolution_step(input, last_encoder_state)
        return self.pooling(input, hidden, context, last_encoder_state, z, f)

    
    def convolution_step(self, input, last_encoder_state):
        # Assumes inputs of shape [minibatch, features, t]
        input = input[:, :, -self.kernel_size:]  # Only use as many time steps
                                                 # as needed for computing
                                                 # the next timestep
        # Prepend padding (otherwise convolution doesn't work
        missing_timesteps = self.kernel_size - input.shape[2]
        if missing_timesteps > 0:
            padding = torch.zeros(input.shape[0], input.shape[1],
                                  missing_timesteps, device=input.device)

            input = torch.cat([padding, input], dim=2)

        training = self.training
        
        enc_proj_z = self.enc_proj_z(last_encoder_state).unsqueeze(2)
        enc_proj_f = self.enc_proj_z(last_encoder_state).unsqueeze(2)
        
        z = torch.tanh(self.z_conv(input) + enc_proj_z)
        self.training = False  # Switch off dropout rescaling
        f = 1 - self.zoneout(1 - self.s(self.f_conv(input) + enc_proj_f))
        self.training = training
        return input, z, f


    def f_pooling(self, input, h, c, last_encoder_state, z, f):
        # We can't really benefit from CUDA here
        # since we're only calculating one timestep
        return f * h + (1-f) * z, c


    def fo_pooling(self, input, h, c, last_encoder_state, z, f):
        enc_proj_o = self.enc_proj_o(last_encoder_state).unsqueeze(2)
        o = torch.sigmoid(self.o_conv(input) + enc_proj_o)
        
        # We can't really benefit from CUDA here
        # since we're only calculating one timestep
        c = f * c + (1-f) * z
        h = o*c
        return h, c


    def ifo_pooling(self, input, h, c, last_encoder_state, z, f):
        enc_proj_o = self.enc_proj_o(last_encoder_state).unsqueeze(2)
        enc_proj_i = self.enc_proj_i(last_encoder_state).unsqueeze(2)
            
        o = torch.sigmoid(self.o_conv(input) + enc_proj_o)
        i = torch.sigmoid(self.i_conv(input) + enc_proj_i)
        
        # We can't really benefit from CUDA here
        # since we're only calculating one timestep
        c = f * c + i * z
        h = o*c
        return h, c


class AttentionLayer(QRNNDecoderLayer):
    def __init__(self, *args, **kwargs):
        super(AttentionLayer, self).__init__(*args, **kwargs)
        encoder_hidden_size = kwargs['encoder_hidden_size']  # Need to know
        hidden_dropout = kwargs['hidden_dropout']
        self.k_project = nn.Linear(encoder_hidden_size, self.hidden_size)
        self.c_project = nn.Linear(self.hidden_size, self.hidden_size)
        self.hidden_dropout = nn.Dropout(p=hidden_dropout)
        
        
    def forward(self, input, h, c, last_encoder_state, encoder_final,
                src_mask):
        input, z, f = self.convolution_step(input, last_encoder_state)
        
        enc_proj_o = self.enc_proj_o(last_encoder_state).unsqueeze(2)
        
        o = torch.sigmoid(self.o_conv(input) + enc_proj_o)
        c = f * h + (1-f) * z  # Use to FO-Pooling for attention
        
        alpha = self.attention(c, encoder_final, src_mask)
        k = self.hidden_dropout(torch.matmul(alpha, encoder_final))
        
        c_projected = self.c_project(c.transpose(1, 2))
        k_projected = self.k_project(k)
        
        # Combine context and attention vectors
        h = (k_projected + c_projected).transpose(1, 2)
        # Apply output gate
        h = h*o
        return k, (h, c), alpha
        
    def attention(self, c, encoder_final, src_mask):
        alpha = torch.matmul(c.transpose(1, 2), encoder_final.transpose(1, 2))
        # Mask out padding
        alpha = torch.where(src_mask, alpha, alpha.new_full([1], float('-inf')))
        alpha = torch.softmax(alpha, dim=2)
        return alpha
