import math
import os
import numpy as np
import torch
import torch.nn as nn
import torch.distributions as dist
import torch.nn.functional as F

class ScaledTanh(nn.Module):
    def __init__(self, alpha_init=1.0):
        super(ScaledTanh, self).__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha_init))

    def forward(self, x):
        return torch.tanh(self.alpha * x)
    
class PINN_Net_Adapt(nn.Module):
    def __init__(self, x_dim, n, m):   # n: # of layers; m: # of neurons
        super(PINN_Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(x_dim+1, m),
            ScaledTanh()
        )

        for i in range(n):
            hidden_layer = nn.Sequential(
            nn.Linear(m, m),
            ScaledTanh()
            )
            self.net.append(hidden_layer)

        final_layer = nn.Sequential(
            nn.Linear(m, 1),
            nn.Softplus()              # make sure that outputs are always positive

        )    
        self.net.append(final_layer)

    def forward(self, x, t):
            return self.net(torch.cat([x, t], dim=-1))

class P2INN(nn.Module):
    def __init__(self, x_dim, param_dim, depth, m):
        super(P2INN, self).__init__()   
        self.coord_enc = nn.Sequential(
            nn.Linear(x_dim+1, m),
            nn.Tanh()
        )

        self.param_enc = nn.Sequential(
            nn.Linear(param_dim, m),
            nn.Tanh()
        )

        self.decoder = nn.Sequential(
            nn.Linear(2*m, m),
            nn.Tanh()
        )

        for _ in range(depth):
            hidden_layer_coord = nn.Sequential(
                nn.Linear(m, m),
                nn.Tanh()
            )

            hidden_layer_param = nn.Sequential(
                nn.Linear(m, m),
                nn.Tanh()
            )

            hidden_layer_dec = nn.Sequential(
                nn.Linear(m, m),
                nn.Tanh()
            )

            self.coord_enc.append(hidden_layer_coord)
            self.param_enc.append(hidden_layer_param)
            self.decoder.append(hidden_layer_dec)

        final_layer_coord = nn.Sequential(
            nn.Linear(m, m)
            )
            
        final_layer_param = nn.Sequential(
            nn.Linear(m, m)
            )
        
        final_layer_dec = nn.Sequential(
            nn.Linear(m, 1)
        )
        
        self.param_enc.append(final_layer_param)
        self.coord_enc.append(final_layer_coord)
        self.decoder.append(final_layer_dec)

    def forward(self, x, t, theta):
        h_param = self.param_enc(theta)   
        h_coord = self.coord_enc(torch.cat([x, t], dim=-1))

        z = torch.cat([h_coord, h_param], dim=-1)

        return self.decoder(z)



class PINN_Net(nn.Module):
    def __init__(self, x_dim, n, m, positivity=False):   # n: # of layers; m: # of neurons
        super(PINN_Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(x_dim+1, m),
            nn.Tanh()
        )

        for i in range(n):
            hidden_layer = nn.Sequential(
            nn.Linear(m, m),
            nn.Tanh()
            )
            self.net.append(hidden_layer)
        
        final_layer = nn.Sequential(
            nn.Linear(m, 1)
        )
        if positivity:
            final_layer.append(nn.Softplus())
            
        self.net.append(final_layer)

    def forward(self, x, t):
            return self.net(torch.cat([x, t], dim=-1))
    
class MetaPINN(nn.Module):
    def __init__(self, n_layers=5, n_hidden=70, theta_dim=1, x_dim=1):
        super(MetaPINN, self).__init__()
        input_dim = x_dim + 1 + theta_dim  # x + t + theta
        layers = [nn.Linear(input_dim, n_hidden), nn.Tanh()]
        for _ in range(n_layers):
            layers += [nn.Linear(n_hidden, n_hidden), nn.Tanh()]
        layers += [nn.Linear(n_hidden, 1), nn.Softplus()]  # or remove Softplus if not needed
        self.net = nn.Sequential(*layers)

    def forward(self, x, t, theta):
        # Ensure all shapes match
        if theta.dim() == 1:
            theta = theta.view(1, 1).expand(x.size(0), 1)  # broadcast θ across batch
        elif theta.shape[0] == 1 and x.shape[0] > 1:
            theta = theta.expand(x.shape[0], theta.shape[1])  # single θ for entire batch

        input_tensor = torch.cat([x, t, theta], dim=-1)

        return self.net(input_tensor)    
    
class FourierFeatureLayer(nn.Module):
    def __init__(self, input_dim, num_frequencies=16, scale=10.0):
        super(FourierFeatureLayer, self).__init__()
        self.B = nn.Parameter(
            torch.randn(num_frequencies, input_dim) * scale,
            requires_grad=False  # fixed basis
        )

    def forward(self, x):
        # x shape: (batch_size, input_dim)
        x_proj = 2 * np.pi * x @ self.B.T  # shape: (batch_size, num_frequencies)
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class PINN_Net_Fourier(nn.Module):
    def __init__(self, x_dim, n, m, num_frequencies=16, fourier_scale=10.0):
        super(PINN_Net_Fourier, self).__init__()

        self.ff_layer = FourierFeatureLayer(x_dim + 1, num_frequencies, scale=fourier_scale)
        ff_dim = 2 * num_frequencies  # sin & cos

        self.net = nn.Sequential(
            nn.Linear(ff_dim, m),
            nn.Tanh()
        )

        for _ in range(n):
            self.net.append(nn.Linear(m, m))
            self.net.append(nn.Tanh())

        self.net.append(nn.Linear(m, 1))
        self.net.append(nn.Softplus())  # Enforce positivity

    def forward(self, x, t):
        xt = torch.cat([x, t], dim=-1)  # shape (batch_size, x_dim + 1)
        ff_xt = self.ff_layer(xt)      # apply Fourier feature transform
        return self.net(ff_xt)        
