import torch
import torch.nn as nn
import torch.nn.init as init

def common_init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
        # consider also xavier_uniform_, kaiming_uniform_ , orthogonal_
    elif type(m) == nn.Conv2d or type(m) == nn.Conv3d:
        nn.init.kaiming_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif type(m) in [nn.LSTM, nn.RNN, nn.GRU]:
        nn.init.orthogonal_(m.weight_hh_l0)
        nn.init.xavier_uniform_(m.weight_ih_l0)
        nn.init.zeros_(m.bias_hh_l0)
        nn.init.zeros_(m.bias_ih_l0)