import math
from .flows_ext import *
import torch
import torch.nn as nn
import torch.nn.functional as F


class ReLUNet(nn.Module):
    def __init__(self, hidden_layers, hidden_units, in_features, out_features):
        super(ReLUNet, self).__init__()

        self.hidden_layers = hidden_layers
        self.in_layer = nn.Linear(in_features, hidden_units, bias=True)
        self.out_layer = nn.Linear(hidden_units, out_features, bias=True)
        for i in range(hidden_layers):
            name = 'cell{}'.format(i)
            cell = nn.Linear(hidden_units, hidden_units, bias=True)
            setattr(self, name, cell)

    def reset_parameters(self):
        self.in_layer.reset_parameters()
        self.out_layer.reset_parameters()
        for i in range(self.hidden_layers):
            name = 'cell{}'.format(i)
            getattr(self, name).reset_parameters()

    def init_identity(self):
        self.in_layer.weight.data.zero_()
        self.in_layer.bias.data.zero_()
        self.out_layer.weight.data.zero_()
        self.out_layer.bias.data.zero_()
        for i in range(self.hidden_layers):
            name = 'cell{}'.format(i)
            getattr(self, name).weight.data.zero_()
            getattr(self, name).bias.data.zero_()

    def forward(self, input):
        """
        input: (batch_size, seq_length, in_features)
        output: (batch_size, seq_length, out_features)
        """
        h = self.in_layer(input)
        h = F.relu(h)
        for i in range(self.hidden_layers):
            name = 'cell{}'.format(i)
            h = getattr(self, name)(h)
            h = F.relu(h)
        return self.out_layer(h)


class NICETrans(nn.Module):
    def __init__(self,
                 couple_layers,
                 cell_layers,
                 hidden_units,
                 features):
        super(NICETrans, self).__init__()

        self.couple_layers = couple_layers
        masks = torch.zeros(couple_layers, features)

        for i in range(couple_layers):
            name = 'cell{}'.format(i)
            cell = ReLUNet(cell_layers, hidden_units, features, features) # with masking, the input dim is the full
            # feature dim
            setattr(self, name, cell)
            if i % 2 == 0:
                masks[i, :features//2] = 1
            else:
                masks[i, features//2:] = 1
        self.register_buffer('jacobian_loss', torch.zeros(1))
        self.register_buffer('masks', masks)
        # self.init_identity()

    def reset_parameters(self):
        for i in range(self.couple_layers):
            name = 'cell{}'.format(i)
            getattr(self, name).reset_parameters()

    def init_identity(self):
        for i in range(self.couple_layers):
            name = 'cell{}'.format(i)
            getattr(self, name).init_identity()


    def forward(self, input):
        """
        input: (seq_length, batch_size, features)
        h: (seq_length, batch_size, features)
        """

        # For NICE it is a constant
        # jacobian_loss = torch.zeros(1, requires_grad=False)
        # print(input.size())
        ep_size = input.size()
        features = ep_size[-1]
        # h = odd_input
        h = input
        for i in range(self.couple_layers):
            name = 'cell{}'.format(i)
            h_out = getattr(self, name)(h*self.masks[i])
            h = self.masks[i]*h + (h_out + h) * (1-self.masks[i])
        return h, self.jacobian_loss

# Real NVP
# https://github.com/ars-ashuha/real-nvp-pytorch/blob/master/real-nvp-pytorch.ipynb

# nets = lambda: nn.Sequential(nn.Linear(2, 256), nn.LeakyReLU(), nn.Linear(256, 256), nn.LeakyReLU(), nn.Linear(256, 2), nn.Tanh())
# nett = lambda: nn.Sequential(nn.Linear(2, 256), nn.LeakyReLU(), nn.Linear(256, 256), nn.LeakyReLU(), nn.Linear(256, 2))

class RealNVP(nn.Module):
    def __init__(self,
                 couple_layers,
                 cell_layers,
                 hidden_units,
                 features):
        super(RealNVP, self).__init__()

        self.couple_layers = couple_layers
        masks = torch.zeros(couple_layers, features)

        for i in range(couple_layers):
            s_name = 'netS{}'.format(i)
            s_cell = ReLUNet(cell_layers, hidden_units, features, features)  # with masking, the input dim is the full
            # feature dim
            setattr(self, s_name, s_cell)
            t_name = 'netT{}'.format(i)
            t_cell = ReLUNet(cell_layers, hidden_units, features, features)  # with masking, the input dim is the full
            # feature dim
            setattr(self, t_name, t_cell)
            if i % 2 == 0:
                masks[i, :features // 2] = 1
            else:
                masks[i, features // 2:] = 1
        self.register_buffer('masks', masks)

    def f(self, x):
        log_det_J, h = x.new_zeros(x.shape[0]), x
        for i in range(self.couple_layers):
            h_in = self.mask[i] * h
            s_name, t_name = 'netS{}'.format(i), 'netT{}'.format(i)
            s = getattr(self, s_name)(h_in) * (1 - self.mask[i])
            t = getattr(self, t_name)(h_in) * (1 - self.mask[i])
            h = (1 - self.mask[i]) * ( h * torch.exp(s) + t) + h_in
            log_det_J += s.sum(dim=1) # tricky. the determinant is exponetial, thus making the abs useless
        return h, log_det_J

    def forward(self, x):
        return self.f(x) # h, logdet

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

class OriginalNICETrans(nn.Module):
    def __init__(self,
                 couple_layers,
                 cell_layers,
                 hidden_units,
                 features,
                 device='cuda'):
        super(OriginalNICETrans, self).__init__()

        self.device = device
        self.couple_layers = couple_layers

        for i in range(couple_layers):
            name = 'cell{}'.format(i)
            cell = ReLUNet(cell_layers, hidden_units, features//2, features//2)
            setattr(self, name, cell)

    def reset_parameters(self):
        for i in range(self.couple_layers):
            name = 'cell{}'.format(i)
            getattr(self, name).reset_parameters()

    def init_identity(self):
        for i in range(self.couple_layers):
            name = 'cell{}'.format(i)
            getattr(self, name).init_identity()


    def forward(self, input):
        """
        input: (seq_length, batch_size, features)
        h: (seq_length, batch_size, features)
        """

        # For NICE it is a constant
        jacobian_loss = torch.zeros(1, device=self.device,
                                    requires_grad=False)

        ep_size = input.size()
        features = ep_size[-1]
        # h = odd_input
        h = input
        for i in range(self.couple_layers):
            name = 'cell{}'.format(i)
            h1, h2 = torch.split(h, features//2, dim=-1)
            if i%2 == 0:
                h = torch.cat((h1, h2 + getattr(self, name)(h1)), dim=-1)
            else:
                h = torch.cat((h1 + getattr(self, name)(h2), h2), dim=-1)
        return h, jacobian_loss

class FLOWDist(torch.nn.Module):
    def __init__(self, inverse_function, mu, sigma):
        super(FLOWDist, self).__init__()
        self.mu = mu
        self.sigma = sigma
        self.base_dist = torch.distributions.Independent(torch.distributions.Normal(self.mu, self.sigma), 1)
        self.inverse_f = inverse_function
        self.dropout = torch.nn.Dropout(0)
        self.sim_penalty = 0

    def forward(self, inputs):
        inputs_shape = inputs.shape
        inputs = inputs.reshape(-1, inputs_shape[-1]) # deal with high rank tensors
        u, log_jacob = self.inverse_f(inputs)

        self.sim_penalty = F.pairwise_distance(u, inputs).mean()

        return u, log_jacob


    def log_prob(self, inputs):
        inputs_shape = inputs.shape
        inputs = self.dropout(inputs)
        u, log_jacob = self.forward(inputs)
        log_probs = self.base_dist.log_prob(u.unsqueeze(-2))
        det_prob = log_probs + log_jacob
        return det_prob.reshape(*inputs_shape[:-1], -1)

    def reset_sim_penalty(self):
        self.sim_penalty = 0