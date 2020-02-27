import torch
import torch.nn as nn
from srl.srl.networks import (FCNEncoderVAE, FCNDecoderVAE, 
                              FullyConvEncoderVAE, FullyConvDecoderVAE,
                              LGSSM, RNNAlpha)
import numpy as np

def to_img(x, shape):
    assert len(shape) == 2
    sig = nn.Sigmoid()
    x = sig(x)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, *shape)
    return x

def load_models(path, args, mode='eval', device='cuda:0'):
    print("Loading models in path: ", path)
    obs_flatten_dim = int(np.product(args.dim_x))  

    if args.non_linearity=="relu":
        nl = nn.ReLU()
    elif args.non_linearity=="elu":
        nl = nn.ELU()
    else:
        raise NotImplementedError()

    if args.measurement_net == "fcn":
        enc = FCNEncoderVAE(dim_in=obs_flatten_dim,
                            dim_out=args.dim_a,
                            bn=args.use_batch_norm,
                            drop=args.use_dropout,
                            nl=nl,
                            hidden_size=args.fc_hidden_size,
                            stochastic=True).to(device=device)
    elif args.measurement_net == "cnn":
        if args.measurement_uncertainty == 'learn_VAE':
            extra_scalars = 0
        elif args.measurement_uncertainty == 'learn_separate':
            extra_scalars = args.dim_a
        elif args.measurement_uncertainty == 'constant':
            extra_scalars = 0
        elif args.measurement_uncertainty == 'scale':
            extra_scalars = 1
        else:
            raise NotImplementedError()
            
        enc = FullyConvEncoderVAE(input=1,
                                    latent_size=args.dim_a,
                                    bn=args.use_batch_norm,
                                    drop=args.use_dropout,
                                    extra_scalars=extra_scalars,
                                    img_dim=str(args.dim_x[1]),
                                    nl=nl,
                                    stochastic=True).to(device=device)
    else:
        raise NotImplementedError()
        
    try:
        enc.load_state_dict(torch.load(path + "/enc.pth", map_location=device))
        if mode == 'eval':
            enc.eval()
        elif mode == 'train':
            enc.train()
        else:
            raise NotImplementedError()
    except Exception as e: 
        print(e)            
        
    output_nl = None if args.use_binary_ce else nn.Sigmoid()
    
    if args.measurement_net == "fcn":
        dec = FCNDecoderVAE(dim_in=args.dim_a,
                            dim_out=args.dim_x,
                            bn=args.use_batch_norm,
                            drop=args.use_dropout,
                            nl=nl,
                            output_nl=output_nl,
                            hidden_size=args.fc_hidden_size).to(device=device)
    elif args.measurement_net == "cnn":
        dec = FullyConvDecoderVAE(input=1,
                                  latent_size=args.dim_a,
                                  bn=args.use_batch_norm,
                                  drop=args.use_dropout,
                                  img_dim=str(args.dim_x[1]),
                                  nl=nl,
                                  output_nl=output_nl).to(device=device)
    else:
        raise NotImplementedError()
        
    try:
        dec.load_state_dict(torch.load(path + "/dec.pth", map_location=device))
        if mode == 'eval':
            dec.eval()
        elif mode == 'train':
            dec.train()
        else:
            raise NotImplementedError()
    except Exception as e: 
        print(e)            

    # LGSSM and dynamic parameter network
    alpha_net = RNNAlpha(input_size=args.dim_alpha,
                                hidden_size=args.alpha_hidden_size,
                                bidirectional=args.use_bidirectional,
                                net_type=args.alpha_net,
                                K=args.k)
    lgssm = LGSSM(dim_z=args.dim_z,
                    dim_a=args.dim_a,
                    dim_u=args.dim_u,
                    alpha_net=alpha_net,
                    K=args.k,
                    transition_noise=args.transition_noise,
                    emission_noise=args.emission_noise,
                    device=device).to(device=device)    
        
    try:
        lgssm.load_state_dict(torch.load(path + "/lgssm.pth", map_location=device))
        if mode == 'eval':
            lgssm.eval()
        elif mode == 'train':
            lgssm.train()
        else:
            raise NotImplementedError()
    except Exception as e: 
        print(e)             
    
    return enc, dec, lgssm
