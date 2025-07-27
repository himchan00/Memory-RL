import torch
import torch.nn as nn
import torchkit.pytorch_utils as ptu


class Hist(nn.Module):
    name = "hist"

    def __init__(self, input_size, hidden_size, n_layer, activation="relu", **kwargs):
        """
        hyp_emb: If true, use hyperbolic embedding for the history representation
        """
        super().__init__()
        l_layer = [nn.Linear(in_features=input_size, out_features=hidden_size)]
        for _ in range(n_layer-1):
            l_layer.append(get_activation(activation))
            l_layer.append(nn.Linear(in_features=hidden_size, out_features=hidden_size))
        
        self.encoder = nn.Sequential(*l_layer)
        self.hidden_size = hidden_size
        self.num_layers = n_layer


    def forward(self, inputs, h_0):
        """
        inputs: (T, B, input_dim)
        h_0: (1, B, hidden_size)
        return
        output: (T, B, hidden_size)
        h_n: (1, B, hidden_size)
        """
        z = self.encoder(inputs)
        output = torch.cumsum(z, dim = 0) + h_0

        return output, output[-1].unsqueeze(0)

    def get_zero_internal_state(self, batch_size=1):
        return ptu.zeros((1, batch_size, self.hidden_size)).float()



def get_activation(s_act):
    if s_act == 'relu':
        return nn.ReLU(inplace=True)
    elif s_act == 'sigmoid':
        return nn.Sigmoid()
    elif s_act == 'softplus':
        return nn.Softplus()
    elif s_act == 'linear':
        return None
    elif s_act == 'tanh':
        return nn.Tanh()
    elif s_act == 'leakyrelu':
        return nn.LeakyReLU(0.2, inplace=True)
    elif s_act == 'softmax':
        return nn.Softmax(dim=1)
    elif s_act == 'swish':
        return nn.SiLU(inplace=True)
    else:
        raise ValueError(f'Unexpected activation: {s_act}')