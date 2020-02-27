import _pickle as pickle
import json
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as tdist

def set_seed_torch(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    random.seed(seed)

seed = 0
set_seed_torch(seed)

def _init_fn(worker_id):
    np.random.seed(int(seed))

from pprint import pprint
from collections import OrderedDict
from datetime import datetime
from tensorboardX import SummaryWriter
from torchvision import transforms
from torchvision.utils import make_grid, save_image
from torch.distributions import normal
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from args.parser import parse_training_args

from srl.srl.learning_utils.weight_initializations import common_init_weights
from srl.srl.learning_utils.learning_rate_schedulers import ExponentialScheduler

from srl.srl.datasets import DatasetUnsupervisedCached, DatasetRealLifeCache
from srl.srl.networks import (FCNEncoderVAE,
                              FCNDecoderVAE,
                              LGSSM,
                              FullyConvEncoderVAE,
                              FullyConvDecoderVAE,
                              RNNAlpha)
from srl.srl.transforms import (Dropped,
                                DropScalarFeature,
                                GaussianNoise,
                                Normalize,
                                Obstruct,
                                Reshape)

def loop(args):
    assert 0 <= args.opt_vae_epochs <= args.opt_vae_kf_epochs <= args.n_epoch
    device = torch.device(args.device)
    torch.backends.cudnn.deterministic = args.cudnn_deterministic
    torch.backends.cudnn.benchmark = args.cudnn_benchmark

    checkpoint_epoch = 2048
    # Save directory
    if not args.debug:
        time_tag = datetime.strftime(datetime.now(), '%m-%d-%y_%H:%M:%S')
        data_dir = args.storage_base_path + time_tag + '_' + args.comment
        os.makedirs(data_dir, exist_ok=True)

        if args.n_epoch > checkpoint_epoch:
            checkpoint_dir = os.path.join(data_dir, "checkpoints")
            os.makedirs(checkpoint_dir, exist_ok=True)

        args.__dict__ = OrderedDict(sorted(args.__dict__.items(), key=lambda t: t[0]))
        with open(data_dir + '/hyperparameters.txt', 'w') as f:
            json.dump(args.__dict__, f, indent=2)
        writer = SummaryWriter(logdir=data_dir)

    obs_flatten_dim = int(np.product(args.dim_x))

    if args.non_linearity=="relu":
        nl = nn.ReLU()
    elif args.non_linearity=="elu":
        nl = nn.ELU()
    else:
        raise NotImplementedError()

    output_nl = None if args.use_binary_ce else nn.Sigmoid()
 
    if args.measurement_net == 'fcn':
        enc = FCNEncoderVAE(dim_in=obs_flatten_dim,
                            dim_out=args.dim_a,
                            bn=args.use_batch_norm,
                            drop=args.use_dropout,
                            nl=nl,
                            hidden_size=args.fc_hidden_size,
                            stochastic=True).to(device=device)
        dec = FCNDecoderVAE(dim_in=args.dim_a,
                            dim_out=args.dim_x,
                            bn=args.use_batch_norm,
                            drop=args.use_dropout,
                            nl=nl,
                            output_nl=output_nl,
                            hidden_size=args.fc_hidden_size).to(device=device)
    elif args.measurement_net == 'cnn':
        if args.measurement_uncertainty == 'learn_VAE':
            extra_scalars = 0
            extra_scalars_conc = 0
        elif args.measurement_uncertainty == 'learn_separate':
            extra_scalars = args.dim_a
            extra_scalars_conc = 0
        elif args.measurement_uncertainty == 'learn_separate_conc':
            extra_scalars = 0
            extra_scalars_conc = args.dim_a
        elif args.measurement_uncertainty == 'constant':
            extra_scalars = 0
            extra_scalars_conc = 0
        elif args.measurement_uncertainty == 'scale':
            extra_scalars = 1
            extra_scalars_conc = 0
        elif args.measurement_uncertainty == 'feature':
            extra_scalars = 0
            extra_scalars_conc = 0
            R_net = FCNEncoderVAE(dim_in=args.dim_a,
                                    dim_out=args.dim_a,
                                    bn=args.use_batch_norm,
                                    drop=args.use_dropout,
                                    nl=nl,
                                    hidden_size=args.fc_hidden_size,
                                    stochastic=False).to(device=device)
        else:
            raise NotImplementedError()

        enc = FullyConvEncoderVAE(input=1,
                                    latent_size=args.dim_a,
                                    bn=args.use_batch_norm,
                                    drop=args.use_dropout,
                                    nl=nl,
                                    img_dim=str(args.dim_x[1]),
                                    extra_scalars=extra_scalars,
                                    extra_scalars_conc=extra_scalars_conc,
                                    stochastic=True).to(device=device)
        dec = FullyConvDecoderVAE(input=1,
                                    latent_size=args.dim_a,
                                    bn=args.use_batch_norm,
                                    img_dim=str(args.dim_x[1]),
                                    drop=args.use_dropout,
                                    nl=nl,
                                    output_nl=output_nl).to(device=device)

    if args.measurement_uncertainty == 'feature':             
        uncertainty_params = list(R_net.parameters())
    else:
        uncertainty_params = []

    # LGSSM and dynamic parameter network
    alpha_network = RNNAlpha(input_size=args.dim_alpha,
                                hidden_size=args.alpha_hidden_size,
                                bidirectional=args.use_bidirectional,
                                net_type=args.alpha_net,
                                K=args.k)
    lgssm = LGSSM(dim_z=args.dim_z,
                    dim_a=args.dim_a,
                    dim_u=args.dim_u,
                    alpha_net=alpha_network,
                    K=args.k,
                    init_cov=args.init_cov,
                    transition_noise=args.transition_noise,
                    emission_noise=args.emission_noise,
                    device=device).to(device=device)    
    dynamic_params = [lgssm.A, lgssm.B, lgssm.C]


    initial_params = [lgssm.z_n1]
    
    if args.opt == "adam":
        opt_vae = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()),
                                    lr=args.lr, betas=(args.beta1, args.beta2), 
                                    weight_decay=args.weight_decay)
        opt_vae_kf = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()) + 
                                        uncertainty_params +
                                        dynamic_params + initial_params, 
                                        lr=args.lr, betas=(args.beta1, args.beta2), 
                                        weight_decay=args.weight_decay)
        opt_all = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()) + 
                                    uncertainty_params + list(lgssm.parameters()), 
                                    lr=args.lr, betas=(args.beta1, args.beta2), 
                                    weight_decay=args.weight_decay)
    elif args.opt == "sgd":
        opt_vae = torch.optim.SGD(list(enc.parameters()) + list(dec.parameters()), 
                                    lr=args.lr, momentum=0.9, nesterov=True)
        opt_vae_kf = torch.optim.SGD(list(enc.parameters()) + list(dec.parameters()) + 
                                        uncertainty_params +
                                        dynamic_params + initial_params, 
                                        lr=args.lr, momentum=0.9, nesterov=True)
        opt_all = torch.optim.SGD(list(enc.parameters()) + list(dec.parameters()) + 
                                    uncertainty_params + list(lgssm.parameters()), 
                                    lr=args.lr, momentum=0.9, nesterov=True)
    elif args.opt == "adadelta":
        opt_vae = torch.optim.Adadelta(list(enc.parameters()) + list(dec.parameters()))
        opt_vae_kf = torch.optim.Adadelta(list(enc.parameters()) + list(dec.parameters()) + uncertainty_params +
                                            dynamic_params + initial_params)
        opt_all = torch.optim.Adadelta(list(enc.parameters()) + list(dec.parameters()) + uncertainty_params +
                                        list(lgssm.parameters()))
    elif args.opt == "rmsprop":
        opt_vae = torch.optim.RMSprop(list(enc.parameters()) + list(dec.parameters()), 
                                    lr=args.lr, momentum=0.9)
        opt_vae_kf = torch.optim.RMSprop(list(enc.parameters()) + list(dec.parameters()) + uncertainty_params +
                                        dynamic_params + initial_params, 
                                        lr=args.lr, momentum=0.9)
        opt_all = torch.optim.RMSprop(list(enc.parameters()) + list(dec.parameters()) +  uncertainty_params +
                                    list(lgssm.parameters()), 
                                    lr=args.lr, momentum=0.9)
    else:
        raise NotImplementedError()

    if args.weight_init == 'custom':
        enc.apply(common_init_weights)
        dec.apply(common_init_weights)
        lgssm.apply(common_init_weights)
        if args.measurement_uncertainty == 'feature': 
            R_net.apply(common_init_weights)

    if args.scheduler == "exponential":
        lr_scheduler_vae = ExponentialScheduler(opt_vae, lr_decay=0.85, lr_decay_frequency=750, min_lr=3e-6)
        lr_scheduler_vae_kf = ExponentialScheduler(opt_vae_kf, lr_decay=0.85, lr_decay_frequency=750, min_lr=3e-6)
        lr_scheduler_all = ExponentialScheduler(opt_all, lr_decay=0.85, lr_decay_frequency=750, min_lr=3e-6)

    # Loss functions
    if args.use_binary_ce:
        loss_REC = nn.BCEWithLogitsLoss(reduction='none').to(device=device)
    else:
        loss_REC = nn.MSELoss(reduction='none').to(device=device)

    if args.task == "pendulum64":
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            Normalize(mean=0.27, var=1.0 - 0.27) # 64x64
            ])
    elif args.task == "real_life_reacher":
        transform = transforms.Compose([
            DropScalarFeature(scalar_feature_dim=8),
            Reshape((15, 1, args.dim_x[1], args.dim_x[2]))
            ])
    else:
        raise NotImplementedError()

    if args.resume_training_path != "none":
        load_path = [x[0] for x in os.walk(args.resume_training_path)][1]
        # load weights
        lgssm.load_state_dict(torch.load(os.path.join(load_path, "lgssm.pth"), map_location=device))
        enc.load_state_dict(torch.load(os.path.join(load_path, "enc.pth"), map_location=device))
        dec.load_state_dict(torch.load(os.path.join(load_path, "dec.pth"), map_location=device))
        if args.measurement_uncertainty == 'feature':  
            R_net.load_state_dict(torch.load(os.path.join(load_path, "R_net.pth"), map_location=device))

        # load optimizers
        opt_vae.load_state_dict(torch.load(os.path.join(load_path, "opt_vae.pth"), map_location=device))
        opt_vae_kf.load_state_dict(torch.load(os.path.join(load_path, "opt_vae_kf.pth"), map_location=device))
        opt_all.load_state_dict(torch.load(os.path.join(load_path, "opt_all.pth"), map_location=device))

        with open(os.path.join(load_path, "training_info.pkl"), "rb") as f:
            info = pickle.load(f)
            ini_epoch = info["epoch"]
            train_annealing_counter = info["train_annealing_counter"]
            val_annealing_counter = info["val_annealing_counter"]
            train_indices = info["train_indices"]
            val_indices = info["val_indices"]
            transform = info["transform"]
            if args.scheduler != 'none':
                lr_scheduler_vae = info["lr_scheduler_vae"]
                lr_scheduler_vae_kf = info["lr_scheduler_vae_kf"]
                lr_scheduler_all = info["lr_scheduler_all"]

    # Dataset
    if args.task == "real_life_reacher":
        dataset = DatasetRealLifeCache(args.dataset,
                                       transform=transform)
    else:
        dataset = DatasetUnsupervisedCached(args.dataset,
                                            transform=transform,
                                            render_h=args.dim_x[1],
                                            render_w=args.dim_x[2])
    if args.resume_training_path == "none":
        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(args.val_split * dataset_size))
        train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    
    train_loader = DataLoader(dataset,
                              batch_size=args.n_batch,
                              num_workers=args.n_worker,
                              sampler=train_sampler,
                              worker_init_fn=_init_fn)
    val_loader = DataLoader(dataset,
                            batch_size=args.n_batch,
                            num_workers=args.n_worker,
                            sampler=valid_sampler,
                            worker_init_fn=_init_fn)

    train_mini_batches = len(train_loader)
    val_mini_batches = len(val_loader)
    annealing_den_train = float(args.n_annealing_epoch_beta * train_mini_batches)
    annealing_den_val = float(args.n_annealing_epoch_beta * val_mini_batches)

    def kvae(epoch, annealing_counter, opt=None, lr_scheduler=None):
        """Training code for unimodal model."""
        if opt:
            enc.train()
            dec.train()
            lgssm.train()
            loader = train_loader
            
            if epoch < args.n_annealing_epoch_beta:
                annealing_factor_beta = min(1., annealing_counter / annealing_den_train)
                one_to_zero = 1. - (annealing_counter / annealing_den_train)
                annealing_factor_beta = max(1., 99.0 * one_to_zero + 1)
            else:
                annealing_factor_beta = 1.
        else:
            enc.eval()
            dec.eval()
            lgssm.eval()
            loader = val_loader

            if epoch < args.n_annealing_epoch_beta:
                annealing_factor_beta = min(1., annealing_counter / annealing_den_val)
                one_to_zero = 1. - (annealing_counter / annealing_den_val)
                annealing_factor_beta = max(1., 99.0 * one_to_zero + 1)
            else:
                annealing_factor_beta = 1.
            
        avg_l = []
        for idx, data in enumerate(loader):
            if idx == args.n_example:
                break

            # XXX: all trajectories have same length
            x_full = data['images'].float().to(device=device)
            # Sample random range of traj_len
            s_idx = np.random.randint(x_full.shape[1] - args.traj_len + 1)
            e_idx = s_idx + args.traj_len
            x = x_full[:, s_idx:(e_idx - 1)]
            x_dim = x.shape
            N = x_dim[0]
            T = x_dim[1]

            # reshape to (N * T, 1, height, width) & encode and decode
            x = x.reshape(N * T, *x_dim[2:])

            if args.measurement_uncertainty == 'learn_VAE':
                # Use VAE's covariance as R in LGSSM
                a, a_mu, a_logvar = enc(x)
                a_cov = torch.diag_embed(torch.exp(a_logvar)).reshape(N, T, args.dim_a, args.dim_a)
                a_mu = a_mu.reshape(N, T, args.dim_a)
                R = a_cov  
                s = torch.tensor(1.0, requires_grad=False, device=device)
            elif args.measurement_uncertainty == "learn_separate" or \
                    args.measurement_uncertainty == "learn_separate_conc":
                # Learn noise separately in VAE as R in LGSSM
                a, a_mu, a_logvar, R_logvar = enc(x)
                a_cov = torch.diag_embed(torch.exp(a_logvar)).reshape(N, T, args.dim_a, args.dim_a)
                a_mu = a_mu.reshape(N, T, args.dim_a)
                R = torch.diag_embed(torch.exp(R_logvar)).reshape(N, T, args.dim_a, args.dim_a)
                s = torch.tensor(1.0, requires_grad=False, device=device)
            elif args.measurement_uncertainty == 'constant':
                # Use R as R in LGSSM
                a, a_mu, a_logvar = enc(x)
                a_cov = torch.diag_embed(torch.exp(a_logvar)).reshape(N, T, args.dim_a, args.dim_a)
                a_mu = a_mu.reshape(N, T, args.dim_a)
                R = None
                s = torch.tensor(1.0, requires_grad=False, device=device)
            elif args.measurement_uncertainty == 'scale':
                # Scale R by s in LGSSM
                a, a_mu, a_logvar, s = enc(x)
                a_cov = torch.diag_embed(torch.exp(a_logvar)).reshape(N, T, args.dim_a, args.dim_a)
                a_mu = a_mu.reshape(N, T, args.dim_a)
                R = None
                s = torch.diag_embed(s.repeat(1, args.dim_a)).reshape(N, T, *s.shape[1:])
            elif args.measurement_uncertainty == 'feature':
                # Learn noise separately in VAE as R in LGSSM
                a, a_mu, a_logvar = enc(x)
                a_cov = torch.diag_embed(torch.exp(a_logvar)).reshape(N, T, args.dim_a, args.dim_a)
                a_mu = a_mu.reshape(N, T, args.dim_a)
                R_logvar = R_net(a)
                R = torch.diag_embed(torch.exp(R_logvar)).reshape(N, T, args.dim_a, args.dim_a)
                s = torch.tensor(1.0, requires_grad=False, device=device)
            else:
                raise NotImplementedError()
            
            x_hat = dec(a)
            # Reshape to (N, T, dim_a)
            a = a.reshape(N, T, args.dim_a)
            u = data['actions'].float().to(device=device)[:, (s_idx + 1):e_idx]

            if args.use_stochastic_dynamics:
                # Simulate imperfect dynamics "counterfactually" with noisy control inputs
                noise = torch.empty(u.shape, requires_grad=False, device=device).normal_(0, 0.2)
                u = u + noise
                u = torch.clamp(u, min=-2.0, max=2.0)
            
            backward_states = lgssm.smooth(a, u, s=s, R=R)

            # NLL reconstruction
            loss_rec = loss_REC(x_hat, x)
            loss_rec = loss_rec.view(N, T, -1).sum(-1)

            # Loss KL sampling based
            loss_2 = lgssm.get_prior(backward_states, s=s, R=R)

            # q(a|x)
            mvn_a = tdist.MultivariateNormal(a_mu, covariance_matrix=a_cov) # ((N, T, dim_a), (N, T, dim_a, dim_a))
            loss_3 = mvn_a.log_prob(a)
            loss_KL = loss_3 - loss_2

            # free nats margin
            # loss_KL = torch.max(torch.zeros_like(loss_KL), (loss_KL - args.free_nats))

            loss_rec = torch.sum(loss_rec)
            loss_KL = torch.sum(loss_KL)

            if epoch % 1 == 0 and idx == 0:
                if R is None:
                    I = torch.eye(args.dim_a, requires_grad=False, device=device)
                    R = lgssm.R.repeat(N, T, 1, 1)
                    R = s.detach() * R + lgssm.eps * I
                pprint({
                    "Reconstruction (per N)": loss_rec.item() / N,
                    "Reconstruction (per N * T)": loss_rec.item() / (N * T),
                    "KL divergence (per N)": loss_KL.item() / N,
                    "KL divergence (per N * T)": loss_KL.item() / (N * T),
                    "R": R[0,0,:,:]
                })

            total_loss = (args.lam_rec * loss_rec + 
                            annealing_factor_beta * args.lam_kl * loss_KL) / N

            avg_l.append((loss_rec.item() + loss_KL.item()) / N)
            
            annealing_counter += 1

            # jointly optimize everything
            if opt:
                opt.zero_grad()
                total_loss.backward()
                # clip for stable RNN training
                if args.measurement_uncertainty == 'feature':  
                    torch.nn.utils.clip_grad_norm_(R_net.parameters(), 0.5)
                torch.nn.utils.clip_grad_norm_(lgssm.parameters(), 0.5)
                torch.nn.utils.clip_grad_norm_(enc.parameters(), 0.5)
                torch.nn.utils.clip_grad_norm_(dec.parameters(), 0.5)
                opt.step()

                if lr_scheduler:
                    lr_scheduler.step()

        avg_loss = sum(avg_l) / len(avg_l)
        return avg_loss, annealing_counter

    # initialize training variables
    if args.resume_training_path == "none":
        ini_epoch = 0
        train_annealing_counter = 0
        val_annealing_counter = 0
    lr_scheduler = None
    opt = opt_vae
    if args.scheduler != 'none':
        lr_scheduler = lr_scheduler_vae
    
    #XXX: Overwrites learning rate even if loading previous optimizer
    for param_group in opt_all.param_groups:
            param_group['lr'] = args.lr

    # training loop
    try:
        for epoch in range(ini_epoch, ini_epoch + args.n_epoch):
            tic = time.time()
            if epoch >= args.opt_vae_kf_epochs:
                opt = opt_all
                if args.scheduler != 'none':
                    lr_scheduler = lr_scheduler_all
            elif epoch >= args.opt_vae_epochs:
                opt = opt_vae_kf
                if args.scheduler != 'none':
                    lr_scheduler = lr_scheduler_vae_kf

            avg_train_loss, train_annealing_counter = kvae(epoch, train_annealing_counter, opt, lr_scheduler)

            if args.val_split > 0:
                with torch.no_grad():
                    avg_val_loss, val_annealing_counter = kvae(epoch, val_annealing_counter)
            else:
                avg_val_loss = 0
            epoch_time = time.time() - tic

            print("Epoch {}/{}: Avg train loss: {}, Avg val loss: {}, Time per epoch: {}"
                    .format(epoch + 1, ini_epoch + args.n_epoch, avg_train_loss, avg_val_loss, epoch_time))
            if not args.debug:
                writer.add_scalars("Loss", 
                                    {'train': avg_train_loss, 'val': avg_val_loss}, epoch)

            if (epoch + 1) % checkpoint_epoch == 0:
                checkpoint_i_path = os.path.join(checkpoint_dir, str((epoch + 1) // checkpoint_epoch))
                os.makedirs(checkpoint_i_path, exist_ok=True)

                # Save models
                torch.save(lgssm.state_dict(), checkpoint_i_path + '/lgssm.pth')
                torch.save(enc.state_dict(), checkpoint_i_path + '/enc.pth')
                torch.save(dec.state_dict(), checkpoint_i_path + '/dec.pth')
                if args.measurement_uncertainty == 'feature':  
                    torch.save(R_net.state_dict(), checkpoint_i_path + '/R_net.pth')
                # Save optimizers
                torch.save(opt_all.state_dict(), checkpoint_i_path + "/opt_all.pth")
                torch.save(opt_vae.state_dict(), checkpoint_i_path + "/opt_vae.pth")
                torch.save(opt_vae_kf.state_dict(), checkpoint_i_path + "/opt_vae_kf.pth")

    finally:
        if not args.debug:
            if not np.isnan(avg_train_loss):
                # Save models
                torch.save(lgssm.state_dict(), data_dir + '/lgssm.pth')
                torch.save(enc.state_dict(), data_dir + '/enc.pth')
                torch.save(dec.state_dict(), data_dir + '/dec.pth')
                if args.measurement_uncertainty == 'feature':  
                    torch.save(R_net.state_dict(), data_dir + '/R_net.pth')
                # Save optimizers
                torch.save(opt_all.state_dict(), data_dir + "/opt_all.pth")
                torch.save(opt_vae.state_dict(), data_dir + "/opt_vae.pth")
                torch.save(opt_vae_kf.state_dict(), data_dir + "/opt_vae_kf.pth")
            writer.close()  

            # Save training information
            with open(os.path.join(data_dir, "training_info.pkl"), "wb") as f:
                info = {"epoch": epoch,
                        "train_annealing_counter": train_annealing_counter,
                        "val_annealing_counter": val_annealing_counter,
                        "train_indices": train_indices,
                        "val_indices": val_indices,
                        "transform": transform}
                if args.scheduler != 'none':
                    info["lr_scheduler_vae"] = lr_scheduler_vae
                    info["lr_scheduler_vae_kf"] = lr_scheduler_vae_kf
                    info["lr_scheduler_all"] = lr_scheduler_all
                pickle.dump(info, f)

def main():
    args = parse_training_args()
    loop(args)

if __name__ == "__main__":
    main()
