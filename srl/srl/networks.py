import numpy as np
import torch
import torch.nn as nn
import torch.distributions as tdist
import time

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)
    

class FullyConvEncoderVAE(nn.Module):
    def __init__(self, input=1, latent_size=12, bn=True, extra_scalars=0, 
                    extra_scalars_conc=0, drop=False, nl=nn.ReLU(), stochastic=True, img_dim="64"):
        super(FullyConvEncoderVAE, self).__init__()    
        self.stochastic = stochastic
        self.layers = nn.ModuleList()
        self.extra_scalars = extra_scalars
        self.extra_scalars_conc = extra_scalars_conc
        self.latent_size = latent_size
        
        self.layers.append(nn.Conv2d(input, 32, 4, stride=2, bias=False))
        if bn: self.layers.append(nn.BatchNorm2d(32, track_running_stats=True))
        if drop: self.layers.append(nn.Dropout(p=0.5))
        self.layers.append(nl)
        
        self.layers.append(nn.Conv2d(32, 64, 4, stride=2, bias=False))
        if bn: self.layers.append(nn.BatchNorm2d(64, track_running_stats=True))
        if drop: self.layers.append(nn.Dropout(p=0.5))
        self.layers.append(nl)
        
        self.layers.append(nn.Conv2d(64, 128, 4, stride=2, bias=False))
        if bn: self.layers.append(nn.BatchNorm2d(128, track_running_stats=True))
        if drop: self.layers.append(nn.Dropout(p=0.5))
        self.layers.append(nl)
        
        self.layers.append(nn.Conv2d(128, 256, 4, stride=2, bias=False))
        if bn: self.layers.append(nn.BatchNorm2d(256, track_running_stats=True))
        if drop: self.layers.append(nn.Dropout(p=0.5))
        self.layers.append(nl)

        if img_dim == "64":
            n_size = 256 * 2 * 2
        elif img_dim == "128":
            n_size = 256 * 6 * 6
        else:
            raise NotImplementedError()

        if self.stochastic:
            self.fc_mu = nn.Linear(n_size, latent_size + extra_scalars_conc)
            self.fc_logvar = nn.Linear(n_size, latent_size)
        else:
            self.fc = nn.Linear(n_size, latent_size)

        if self.extra_scalars > 0:
            self.fc_extra = nn.Sequential(
                nn.Linear(n_size, 1024),
                nn.ELU(),
                nn.Linear(1024, 1024),
                nn.ReLU(),
                nn.Linear(1024, self.extra_scalars),
                nn.ELU(alpha=4)
            )
        self.flatten = Flatten()

    def forward(self, x):
        for i in range(len(self.layers)):
            x = self.layers[i](x)
        x = self.flatten(x)
        if self.stochastic:
            x_mu = self.fc_mu(x)
            mu = x_mu[:, :self.latent_size]
            logvar = self.fc_logvar(x)
            # Reparameterize
            std = torch.exp(logvar / 2.0)
            eps = torch.randn_like(std)
            z = mu + eps * std

            # Extra variables with shared network
            if self.extra_scalars > 0:
                extra_scalars = self.fc_extra(x)
                # return z, mu, logvar, torch.exp(extra_scalars)
                return z, mu, logvar, extra_scalars

            if self.extra_scalars_conc > 0:
                extra_scalars = x_mu[:, self.latent_size:]
                return z, mu, logvar, extra_scalars

            return z, mu, logvar
        else: 
            z = self.fc(x)
            if self.extra_scalar_size > 0:
                extra_scalars = self.fc_extra(x)
                return z, extra_scalars

            if self.extra_scalars_conc > 0:
                extra_scalars = x_mu[self.latent_size:]
                return z, extra_scalars
            return z


class FullyConvDecoderVAE(nn.Module):
    def __init__(self, input=1, latent_size=12, output_nl=nn.Tanh(), bn=True, 
                    drop=False, nl=nn.ReLU(), img_dim="64"):
        super(FullyConvDecoderVAE, self).__init__()
        self.bn = bn
        self.drop = drop
        self.layers = nn.ModuleList()

        if img_dim == "64":
            n_size = 256 * 2 * 2
        elif img_dim == "128":
            n_size = 256 * 6 * 6
        else:
            raise NotImplementedError()

        self.layers.append(nn.ConvTranspose2d(n_size, 128, 5, stride=2, bias=False))
        if bn: self.layers.append(nn.BatchNorm2d(128))
        if drop: self.layers.append(nn.Dropout(p=0.5))
        self.layers.append(nl)

        self.layers.append(nn.ConvTranspose2d(128, 64, 5, stride=2, bias=False))
        if bn: self.layers.append(nn.BatchNorm2d(64, track_running_stats=True))
        if drop: self.layers.append(nn.Dropout(p=0.5))
        self.layers.append(nl)

        if img_dim == "64":
            self.layers.append(nn.ConvTranspose2d(64, 32, 6, stride=2, bias=False))
            if bn: self.layers.append(nn.BatchNorm2d(32, track_running_stats=True))
            if drop: self.layers.append(nn.Dropout(p=0.5))
            self.layers.append(nl)

            self.layers.append(nn.ConvTranspose2d(32, input, 6, stride=2, bias=False))
            if bn: self.layers.append(nn.BatchNorm2d(input, track_running_stats=True))
            if drop: self.layers.append(nn.Dropout(p=0.5))
        elif img_dim == "128":
            self.layers.append(nn.ConvTranspose2d(64, 32, 5, stride=2, bias=False))
            if bn: self.layers.append(nn.BatchNorm2d(32, track_running_stats=True))
            if drop: self.layers.append(nn.Dropout(p=0.5))
            self.layers.append(nl)

            self.layers.append(nn.ConvTranspose2d(32, 16, 6, stride=2, bias=False))
            if bn: self.layers.append(nn.BatchNorm2d(16, track_running_stats=True))
            if drop: self.layers.append(nn.Dropout(p=0.5))

            self.layers.append(nn.ConvTranspose2d(16, input, 6, stride=2, bias=False))
            if bn: self.layers.append(nn.BatchNorm2d(input, track_running_stats=True))
            if drop: self.layers.append(nn.Dropout(p=0.5))
        else:
            raise NotImplementedError()

        if output_nl != None:
            self.layers.append(output_nl)

        self.linear = nn.Linear(latent_size, n_size, bias=False)
        self.batchn = nn.BatchNorm1d(n_size)
        self.dropout = nn.Dropout(p=0.5)
        self.nl = nl
        
    def forward(self, x):
        if self.bn:
            x = self.nl(self.batchn(self.linear(x)))
        elif self.drop:
            x = self.nl(self.dropout(self.linear(x)))
        else:
            x = self.nl(self.linear(x))

        x = x.unsqueeze(-1)
        x = x.unsqueeze(-1)
        for i in range(len(self.layers)):
            x = self.layers[i](x)

        return x


class FCNEncoderVAE(nn.Module):
    def __init__(self, dim_in, dim_out, bn=False, drop=False, nl=nn.ReLU(), hidden_size=800, stochastic=True):
        super(FCNEncoderVAE, self).__init__()
        self.flatten = Flatten()
        self.stochastic = stochastic
        self.bn = bn
        self.layers = nn.ModuleList()

        self.layers.append(torch.nn.Linear(dim_in, hidden_size))
        if bn: self.layers.append(nn.BatchNorm1d(hidden_size, track_running_stats=True))
        if drop: self.layers.append(nn.Dropout(p=0.5))
        self.layers.append(nl)

        self.layers.append(torch.nn.Linear(hidden_size, hidden_size))
        if bn: self.layers.append(nn.BatchNorm1d(hidden_size, track_running_stats=True))
        if drop: self.layers.append(nn.Dropout(p=0.5))
        self.layers.append(nl)

        if stochastic:
            self.layers.append(torch.nn.Linear(hidden_size, 2 * dim_out))
        else:
            self.layers.append(torch.nn.Linear(hidden_size, dim_out))

    def forward(self, x):
        x = self.flatten(x)
        for l in self.layers:
            x = l(x)

        if self.stochastic:
            print(x.shape)
            mu, logvar = x.chunk(2, dim=1)
            # Reparameterize
            std = torch.exp(logvar / 2.0)
            eps = torch.randn_like(std)
            z = mu + eps * std
            return z, mu, logvar
        else: 
            return x


class FCNDecoderVAE(nn.Module):
    def __init__(self, dim_in, dim_out, bn=False, drop=False, nl=nn.ReLU(), output_nl=None, hidden_size=800):
        super(FCNDecoderVAE, self).__init__()
        self.dim_out = dim_out
        self.layers = nn.ModuleList()

        self.layers.append(torch.nn.Linear(dim_in, hidden_size))
        if bn: self.layers.append(nn.BatchNorm1d(hidden_size, track_running_stats=True))
        if drop: self.layers.append(nn.Dropout(p=0.5))
        self.layers.append(nl)

        self.layers.append(torch.nn.Linear(hidden_size, hidden_size))
        if bn: self.layers.append(nn.BatchNorm1d(hidden_size, track_running_stats=True))
        if drop: self.layers.append(nn.Dropout(p=0.5))
        self.layers.append(nl)

        self.layers.append(torch.nn.Linear(hidden_size, int(np.product(dim_out))))
        if output_nl != None:
            self.layers.append(output_nl)
            
    def forward(self, z):
        for l in self.layers:
            z = l(z)
        x = z.view(-1, *self.dim_out)
        return x


class RNNAlpha(nn.Module):
    """
    This class defines the GRU-based or LSTM-based dynamics parameter network alpha from
    https://github.com/simonkamronn/kvae/blob/master/kvae/filter.py.

    Args:
        input_size: Input dimension
        hidden_size: Hidden state dimension
        K: Mixture amount
        layers: Number of layers
        bidirectional: Use bidirectional version
        net_type: Use the LSTM or GRU variation
    """
    def __init__(self, input_size, hidden_size=128, K=1, layers=1, bidirectional=False, net_type="lstm"):
        super(RNNAlpha, self).__init__()
        self.K = K
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.layers = layers
        if net_type == "gru":
            self.rnn = nn.GRU(input_size=input_size, hidden_size=hidden_size, 
                                num_layers=layers, bidirectional=bidirectional)
        elif net_type =="lstm":
            self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, 
                                num_layers=layers, bidirectional=bidirectional)
        if bidirectional:
            self.linear = nn.Linear(in_features=2*hidden_size, out_features=K)
        else:
            self.linear = nn.Linear(in_features=hidden_size, out_features=K)

        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, a, h=None):
        """
        Forward call to produce the alpha mixing weights.

        Args:
            a: pseudo measurements from the VAEs (seq_len, batch_size, dim_a)
            h: hidden state of the LSTM (num_layers * num_directions, batch_size, hidden_size) or None. If None, h is defaulted as 0-tensor
        Returns:
            alpha: mixing vector of dimension K (batch_size, seq_len, K)
        """
        L, N, _ = a.shape
        if h is None:
            x, h = self.rnn(a)
        else:
            x, h = self.rnn(a, h)
        
        if self.bidirectional:
            x = x.reshape(L * N, 2*self.hidden_size) # (seq_len * batch_size, 2 * hidden_size)
        else:
            x = x.reshape(L * N, self.hidden_size) # (seq_len * batch_size, hidden_size)

        x = self.linear(x)
        alpha = self.softmax(x)
        alpha = alpha.reshape(L, N, self.K).transpose(1,0) # (batch_size, seq_len, hidden_size)
        return alpha, h
        

class LGSSM(nn.Module):
    """
    This class defines a Kalman Filter (Linear Gaussian State Space model), possibly with a dynamics parameter
    network alpha which uses a weighed combination of base linear matrices. 
    Based on https://github.com/simonkamronn/kvae/blob/master/kvae/filter.py.
    """
    def __init__(self, dim_z, dim_a, dim_u, alpha_net, device,
                 K=4, init_cov=20.0, transition_noise=1., emission_noise=1., init_kf_matrices=1.):
        super(LGSSM, self).__init__()
        self.dim_z = dim_z
        self.dim_a = dim_a
        self.dim_u = dim_u
        self.device = device

        # initial distribution p(z0)
        self.mu_0 = torch.zeros((dim_z, 1), requires_grad=False, device=device)
        self.Sigma_0 = init_cov * torch.eye(dim_z, requires_grad=False, device=device)

        self.A = nn.Parameter(torch.eye(dim_z, device=device).repeat(K, 1, 1))
        self.B = nn.Parameter(init_kf_matrices * torch.rand((K, dim_z, dim_u), device=device))
        self.C = nn.Parameter(init_kf_matrices * torch.rand((K, dim_a, dim_z), device=device))

        # initial latent state
        self.z_n1 = nn.Parameter(torch.zeros((dim_z, 1), device=device))

        # untrainable uncertainties for now
        self.Q = torch.eye(dim_z, requires_grad=False, device=device) * transition_noise
        self.R = torch.eye(dim_a, requires_grad=False, device=device) * emission_noise
        self.I = torch.eye(dim_z, requires_grad=False, device=device)
        self.eps = 0

        # amount of mixture components
        self.K = K

        # dynamic parameter network
        self.alpha_net = alpha_net.to(device=device)

    def initialize(self, a, u, s=1.0, R=None):
        """
        Initialize state with a window of "T".
        Args:
            a: Initial encoded measurements (batch_size, T, dim_a) e.g. [a_0, a_1, ..., a_(T-1)]
            u: Initial control inputs (batch_size, T, dim_u) e.g. [u_1, u_2, ..., u_T]
            s: (batch_size, T, dim_a, dim_a)
            R: (batch_size, T, dim_a, dim_a)
        Returns:
            mu_i: latest mu filtered @ index T - 1, (batch_size, dim_z, 1)
            Sigma_i: latest Sigma filtered @ index T - 1, (batch_size, dim_z, dim_z)
            alpha_i: latest alpha calculated from index T -1, (batch_size, K)
            h_i: latest hidden state @ index T
        """
        with torch.no_grad():
            backward_states = self.smooth(a, u, s=s, R=R)

            mu_smooth, Sigma_smooth, _, _, _, _, _, alpha_smooth, h_last = backward_states

            # initial state is last filtered state @ index T-1
            mu_i = mu_smooth[:, -1, :, :] # mu_smooth @ index T - 1, (batch_size, dim_z, 1)
            Sigma_i = Sigma_smooth[:, -1, :, :] # Sigma_smooth @ index T - 1, (batch_size, dim_z, dim_z)
            alpha_i = alpha_smooth[:, -1, :] # alpha @ index T calculated from index T -1, (batch_size, K)
            h_i = h_last

            return mu_i, Sigma_i, alpha_i, h_i

    def predict(self, mu_tn1, Sigma_tn1, alpha_t, h_t, u_f):
        """
        Predict or generate "pred_len" future states based on control input "u_f"
        Args:
            mu_tn1: initial mu filtered @ index T - 1 (batch_size, dim_z, 1)
            Sigma_tn1: initial Sigma filtered @ index T - 1, (batch_size, dim_z, dim_z)
            alpha_t: initial alpha T calculated from index T-1, (batch_size, K)
            h_t: initial hidden state @ index T
            u_f: Future control inputs (batch_size, pred_len, dim_u) [u_{T}, u_{T+1}, ..., u_{T+pred_len-1}]
        Returns:
            z: (batch_size, pred_len, dim_z, 1)
            mu: (batch_size, pred_len, dim_z, 1)
            Sigma: (batch_size, pred_len, dim_z, dim_z)
            a: (batch_size, pred_len, dim_a, 1)
            A: (batch_size, pred_len, dim_z, dim_z)
            B: (batch_size, pred_len, dim_z, dim_u)
            C: (batch_size, pred_len, dim_a, dim_z)
        """
        with torch.no_grad():
            batch_size, pred_len = u_f.shape[0], u_f.shape[1]
            # pre-allocate forward states needed
            a = torch.empty((batch_size, pred_len, self.dim_a, 1), device=self.device)
            z = torch.empty((batch_size, pred_len, self.dim_z, 1), device=self.device)
            mu = torch.empty((batch_size, pred_len, self.dim_z, 1), device=self.device)
            Sigma = torch.empty((batch_size, pred_len, self.dim_z, self.dim_z), device=self.device)
            A = torch.empty((batch_size, pred_len, self.dim_z, self.dim_z), device=self.device)
            B = torch.empty((batch_size, pred_len, self.dim_z, self.dim_u), device=self.device)
            C = torch.empty((batch_size, pred_len, self.dim_a, self.dim_z), device=self.device)

            for ii in range(pred_len):
                u_t = u_f[:, ii, :] 

                # mixture of A
                A_t = torch.mm(alpha_t, self.A.reshape(-1, self.dim_z * self.dim_z)) # (bs, k) x (k, dim_z*dim_z) 
                A_t = A_t.reshape(-1, self.dim_z, self.dim_z) # (bs, dim_z, dim_z)
                # mixture of B
                B_t = torch.mm(alpha_t, self.B.reshape(-1, self.dim_z * self.dim_u)) # (bs, k) x (k, dim_z*dim_u) 
                B_t = B_t.reshape(-1, self.dim_z, self.dim_u) # (bs, dim_z, dim_u)
                # mixture of C
                C_t = torch.mm(alpha_t, self.C.reshape(-1, self.dim_a * self.dim_z)) # (bs, k) x (k, dim_a*dim_z) 
                C_t = C_t.reshape(-1, self.dim_a, self.dim_z) # (bs, dim_a, dim_z)

                # prediction
                mu_t = torch.bmm(A_t, mu_tn1) + torch.bmm(B_t, u_t.unsqueeze(-1))
                Sigma_t = torch.bmm(torch.bmm(A_t, Sigma_tn1), A_t.transpose(-1,-2)) + self.Q
                mvn = tdist.MultivariateNormal(torch.squeeze(mu_t), covariance_matrix=Sigma_t)
                z_t_sampled = mvn.sample().unsqueeze(-1) # (bs, dim_z, 1)
                a_pred = torch.bmm(C_t, z_t_sampled)

                # store
                z[:, ii, :, :] = z_t_sampled
                a[:, ii, :, :] = a_pred
                mu[:, ii, :, :] = mu_t
                Sigma[:, ii, :, :] = Sigma_t
                A[:, ii, :, :] = A_t # (0, ..., T)
                B[:, ii, :, :] = B_t # (0, ..., T)
                C[:, ii, :, :] = C_t # (0, ..., T)

                alpha_out, h_out = self.alpha_net(a=z_t_sampled.squeeze(-1).unsqueeze(0), h=h_t)

                # restart
                alpha_t = alpha_out[:, 0, :]
                h_t = h_out
                mu_tn1 = mu_t
                Sigma_tn1 = Sigma_t

            return z, mu, Sigma, a, A, B, C

    def forward(self, a, u, u_f, s=1.0, R=None):
        """
        Predict or generate "pred_len" future states based on control input 
        and an initial history of "T".
        Args:
            a: Initial encoded measurements (batch_size, T, dim_a) e.g. [a_0, a_1, ..., a_(T-1)]
            u: Initial control inputs (batch_size, T, dim_u) e.g. [u_1, u_2, ..., u_T]
            u_f: Future control inputs (batch_size, pred_len, dim_u) [u_{T}, u_{T+1}, ..., u_{T+pred_len-1}]
            s: (batch_size, T, dim_a, dim_a)
            R: (batch_size, T, dim_a, dim_a)
        Returns:
            z: (batch_size, pred_len, dim_z, 1)
            mu: (batch_size, pred_len, dim_z, 1)
            Sigma: (batch_size, pred_len, dim_z, dim_z)
            a: (batch_size, pred_len, dim_a, 1)
            A: (batch_size, pred_len, dim_z, dim_z)
            B: (batch_size, pred_len, dim_z, dim_u)
            C: (batch_size, pred_len, dim_a, dim_z)
        """

        with torch.no_grad():
            mu_tn1, Sigma_tn1, alpha_t, h_t = self.initialize(a, u, s, R)
            z, mu, Sigma, a, A, B, C = self.predict(mu_tn1, Sigma_tn1, alpha_t, h_t, u_f)
        return z, mu, Sigma, a, A, B, C

    def filter_update(self, mu_pred, Sigma_pred, alpha, a, R):
        """
        Filter a single predicted state.
        """
        # mixture of C
        C = torch.mm(alpha, self.C.reshape(-1, self.dim_a * self.dim_z)) # (bs, k) x (k, dim_a*dim_z) 
        C = C.reshape(-1, self.dim_a, self.dim_z) # (bs, dim_a, dim_z)

        # residual
        a_pred = torch.bmm(C, mu_pred) # (bs, dim_a, 1)
        r_t = a.unsqueeze(-1) - a_pred

        # project uncertainty into measurement space
        S_t = torch.bmm(torch.bmm(C, Sigma_pred), C.transpose(-1, -2)) + R # (bs, dim_a, dim_a)
        S_t_inv = torch.inverse(S_t)

        # Kalman gain
        K_t = torch.bmm(torch.bmm(Sigma_pred, C.transpose(-1, -2)), S_t_inv) # (bs, dim_z, dim_a)

        # measurement update using Joseph's form
        mu_filt = mu_pred + torch.bmm(K_t, r_t)
        ImKC = self.I - torch.bmm(K_t, C) # (bs, dim_z, dim_z)
        Sigma_filt = torch.bmm(torch.bmm(ImKC, Sigma_pred), ImKC.transpose(-1,-2)) + \
                    torch.bmm(torch.bmm(K_t, R), K_t.transpose(-1,-2))
        return mu_filt, Sigma_filt, C

    def predict_update(self, mu_filt, Sigma_filt, alpha, u):
        """
        Predict a single filtered state.
        """
        # mixture of A
        A = torch.mm(alpha, self.A.reshape(-1, self.dim_z * self.dim_z)) # (bs, k) x (k, dim_z*dim_z) 
        A = A.reshape(-1, self.dim_z, self.dim_z) # (bs, dim_z, dim_z)

        # mixture of B
        B = torch.mm(alpha, self.B.reshape(-1, self.dim_z * self.dim_u)) # (bs, k) x (k, dim_z*dim_u) 
        B = B.reshape(-1, self.dim_z, self.dim_u) # (bs, dim_z, dim_u)

        # prediction
        mu_pred = torch.bmm(A, mu_filt) + torch.bmm(B, u.unsqueeze(-1))
        Sigma_pred = torch.bmm(torch.bmm(A, Sigma_filt), A.transpose(-1,-2)) + self.Q
        return mu_pred, Sigma_pred, A, B

    def compute_forward_step(self, mu_pred_t, Sigma_pred_t, alpha_t, h_last, a_t, R_t, u_tp1):
        """
        Compute the forward step in the Kalman filter (measurement update then prediction).
        Args:
            mu_pred_t: Previous time step's mean prediction (batch_size, dim_z, 1)
            Sigma_pred_t: Previous time step's covariance prediction (batch_size, dim_z, dim_z)
            alpha_t: Previous time step's alpha (batch_size, K)
            h_last: Previous time step's hidden state
            a_t: Previous time step's measurement (batch_size, dim_a)
            R_t: Previous time step's measurement covariance (batch_size, dim_a, dim_a)
            u_tp1: Current control inputs (batch_size, dim_u)

        Returns:
            mu_filt_t: Previous time step's prediction mean updated with the measurement (batch_size, dim_z, 1)
            Sigma_filt_t: Previous time step's prediction covariance updated with the measurement (batch_size, dim_z, dim_z)
            mu_pred_tp1: Current time step's mean prediction (batch_size, dim_z, 1)
            Sigma_pred_tp1: Current time step's covariance prediction (batch_size, dim_z, dim_z)
            A_tp1: Current time step's transition matrix (batch_size, dim_z, dim_z)
            B_tp1:
            C_t:
            alpha_tp1:
            h_last_tp1:
        """
        mu_filt_t, Sigma_filt_t, C_t = \
            self.filter_update(mu_pred_t, Sigma_pred_t, alpha_t, a_t, R_t)

        mvn = tdist.MultivariateNormal(mu_filt_t.squeeze(-1), covariance_matrix=Sigma_filt_t) # ((batch_size, dim_z), (batch_size, dim_z, dim_z))
        z_t = mvn.rsample() # (batch_size, dim_z)
        z_t = z_t.unsqueeze(0) # (1, batch_size, dim_z)
        alpha_tp1, h_last_tp1 = self.alpha_net(z_t, h_last) # (batch_size, 1, K)
        alpha_tp1 = alpha_tp1[:,0,:]

        mu_pred_tp1, Sigma_pred_tp1, A_tp1, B_tp1 = \
            self.predict_update(mu_filt_t, Sigma_filt_t, alpha_tp1, u_tp1)

        return mu_filt_t, Sigma_filt_t, mu_pred_tp1, Sigma_pred_tp1, A_tp1, B_tp1, C_t, alpha_tp1, h_last_tp1

    def compute_forward(self, a, u, s=1.0, R=None):
        """
        Get forward states based on forward pass.
        Args:
            a: (batch_size, T, dim_a) e.g. [a_0, a_1, ..., a_(T-1)]
            u: (batch_size, T, dim_u) e.g. [u_1, u_2, ..., u_T]
            a_cov: (batch_size, T, dim_a, dim_a) e.g. [a_cov_0, a_cov_1, ..., a_cov_(T-1)]

        Returns:
            forward_states: (batch_size, T, <feature_dim1>, <feature_dim2>)
        """
        batch_size = a.shape[0]
        T = a.shape[1]
        I = torch.eye(self.dim_a, requires_grad=False, device=self.device)
        if R is None:
            R = self.R.repeat(batch_size, T, 1, 1)
        else:
            R = R

        R = s * R + self.eps * I
        # sample initial state
        mu_pred_t = self.mu_0.repeat(batch_size, 1, 1) # (batch_size, dim_z, 1)
        Sigma_pred_t = self.Sigma_0.repeat(batch_size, 1, 1) # (batch_size, dim_z, dim_z)

        # pre-allocate forward states needed
        mu_filt = torch.empty((batch_size, T, self.dim_z, 1)).to(device=self.device)
        Sigma_filt = torch.empty((batch_size, T, self.dim_z, self.dim_z)).to(device=self.device)
        mu_pred = torch.empty((batch_size, T, self.dim_z, 1)).to(device=self.device)
        Sigma_pred = torch.empty((batch_size, T, self.dim_z, self.dim_z)).to(device=self.device)
        alpha = torch.empty((batch_size, T, self.K)).to(device=self.device)
        A = torch.empty((batch_size, T, self.dim_z, self.dim_z)).to(device=self.device)
        B = torch.empty((batch_size, T, self.dim_z, self.dim_u)).to(device=self.device)
        C = torch.empty((batch_size, T, self.dim_a, self.dim_z)).to(device=self.device)

        z_n1 = self.z_n1.repeat(batch_size, 1, 1).squeeze(-1).unsqueeze(0)
        alpha_t, h_last_t = self.alpha_net(z_n1) # (batch_size, 1, K)
        alpha_t = alpha_t[:,0,:]
        
        # single steps (roll-out from index 0 ... T - 1) + prediction at T 
        for tt in range(T):
            mu_filt_t, Sigma_filt_t, mu_pred_tp1, Sigma_pred_tp1, A_tp1, B_tp1, C_t, alpha_tp1, h_last_tp1 = \
                self.compute_forward_step(mu_pred_t=mu_pred_t, Sigma_pred_t=Sigma_pred_t, 
                                            alpha_t=alpha_t, h_last=h_last_t, 
                                            a_t=a[:, tt, :], R_t=R[:, tt, :, :], u_tp1=u[:, tt, :])
            
            # store results
            mu_pred[:, tt, :, :] = mu_pred_tp1
            Sigma_pred[:, tt, :, :] = Sigma_pred_tp1
            mu_filt[:, tt, :, :] = mu_filt_t
            Sigma_filt[:, tt, :, :] = Sigma_filt_t
            alpha[:, tt, :] = alpha_tp1

            A[:, tt, :, :] = A_tp1
            B[:, tt, :, :] = B_tp1
            C[:, tt, :, :] = C_t

            # restart
            mu_pred_t = mu_pred_tp1
            Sigma_pred_t = Sigma_pred_tp1
            alpha_t = alpha_tp1
            h_last_t = h_last_tp1

        forward_states = (mu_pred, Sigma_pred, mu_filt, Sigma_filt, A, B, C, a, u, alpha, h_last_t)
        return forward_states

    def compute_backward_step(self, mu_smooth_tp1, Sigma_smooth_tp1, mu_pred_tp1, Sigma_pred_tp1,
                                mu_filt_t, Sigma_filt_t, A_tp1):
        """
        Compute the backward step in the Kalman smoother.
        Args:
            mu_smooth_tp1: Future time step's smoothed distribution mean (batch_size, dim_z, 1)
            Sigma_smooth_tp1: Future time step's smoothed distribution covariance (batch_size, dim_z, dim_z)
            mu_pred_tp1: (batch_size, dim_z, 1)
            Sigma_pred_tp1: (batch_size, dim_z, dim_z)
            mu_filt_t: (batch_size, dim_z, 1)
            Sigma_filt_t: (batch_size, dim_z, dim_z)
            A_tp1: (batch_size, dim_z, dim_z)

        Returns:
            mu_smooth_t: Previous time step's smoothed distribution mean (batch_size, dim_z, 1)
            Sigma_smooth_t: Previous time step's smoothed distribution covariance (batch_size, dim_z, dim_z)
        """
        Sigma_pred_tp1_inv = torch.inverse(Sigma_pred_tp1)
        J = torch.bmm(torch.bmm(Sigma_filt_t, A_tp1.transpose(-1,-2)), Sigma_pred_tp1_inv)

        mu_smooth_t = mu_filt_t + torch.bmm(J, (mu_smooth_tp1 - mu_pred_tp1))
        JSJt = torch.bmm(torch.bmm(J, (Sigma_smooth_tp1 - Sigma_pred_tp1)), J.transpose(-1, -2))
        Sigma_smooth_t = Sigma_filt_t + JSJt

        return mu_smooth_t, Sigma_smooth_t

    def compute_backward(self, forward_states):
        """
        Get backward states based on smoothing.
        Args:
            forward_states: tuple of (mu_pred, Sigma_pred, mu_filt, Sigma_filt, A, B, C, a, u)
                            (batch_size, T, <feature_dim1>, <feature_dim2>)

        Returns:
            backward_states: tuple of (mu_smooth, Sigma_smooth, A, B, C, a, u)
        """
        mu_pred, Sigma_pred, mu_filt, Sigma_filt, A, B, C, a, u, alpha, h_last = forward_states

        # Number of states
        batch_size = mu_filt.shape[0]
        T = mu_filt.shape[1]

        # pre-allocated smoothing backward states needed
        mu_smooth = torch.empty((batch_size, T, self.dim_z, 1)).to(device=self.device)
        Sigma_smooth = torch.empty((batch_size, T, self.dim_z, self.dim_z)).to(device=self.device)

        # last smoothing. state (T-1) is just the last filtering state used to initialize
        mu_smooth[:, 0, :, :] = mu_filt[:, -1, :, :]
        Sigma_smooth[:, 0, :, :] = Sigma_filt[:, -1, :, :]

        # initial variables for backwards pass 
        mu_smooth_tp1 = mu_filt[:, -1, :, :]
        Sigma_smooth_tp1 = Sigma_filt[:, -1, :, :]

        # discard last time dimension to account for indices, 
        # predictive t=1, ..., T-1
        # filter t=0, ..., T-2
        mu_pred = mu_pred[:, :-1, :, :]
        Sigma_pred = Sigma_pred[:, :-1, :, :]
        mu_filt = mu_filt[:, :-1, :, :]
        Sigma_filt = Sigma_filt[:, :-1, :, :]
        A_backward = A[:, :-1, :, :]

        # Number of states to loop backwards
        Tm1 = T - 1

        # reverse direction along "time" direction
        # predictive t=T-1, ..., 1
        # filter t=T-2, ..., 0
        mu_pred = torch.flip(mu_pred, (1,))
        Sigma_pred = torch.flip(Sigma_pred, (1,))
        mu_filt = torch.flip(mu_filt, (1,))
        Sigma_filt = torch.flip(Sigma_filt, (1,))
        A_backward  = torch.flip(A_backward, (1,))

        for tt in range(Tm1):
            mu_smooth_t, Sigma_smooth_t = self.compute_backward_step(
                mu_smooth_tp1=mu_smooth_tp1,
                Sigma_smooth_tp1=Sigma_smooth_tp1,
                mu_pred_tp1=mu_pred[:, tt, :, :],
                Sigma_pred_tp1=Sigma_pred[:, tt, :, :],
                mu_filt_t=mu_filt[:, tt, :, :],
                Sigma_filt_t=Sigma_filt[:, tt, :, :],
                A_tp1=A_backward[:, tt, :, :]
            )
            mu_smooth_tp1 = mu_smooth_t
            Sigma_smooth_tp1 = Sigma_smooth_t
            mu_smooth[:, tt + 1, :, :] = mu_smooth_t # skip initial which is set as filtering state
            Sigma_smooth[:, tt + 1, :, :] = Sigma_smooth_t

        # reverse direction to orthodox "time" direction
        # smooth t=0, ..., T-1
        mu_smooth = torch.flip(mu_smooth, (1,))
        Sigma_smooth = torch.flip(Sigma_smooth, (1,))

        backward_states = (mu_smooth, Sigma_smooth, A, B, C, a, u, alpha, h_last)
        return backward_states

    def filter(self, a, u, R=None, s=1.0):
            return self.compute_forward(a, u, s=s, R=R)

    def smooth(self, a, u, s=1.0, R=None):
            return self.compute_backward(self.compute_forward(a, u, s=s, R=R))

    def get_prior(self, backward_states, s=1.0, R=None):
        """
        Calculate the prior of a sample for the LGSSM.

        Args:
            backward_states: Smoothed states ((batch_size, T, dim_z, 1), (batch_size, T, dim_z, dim_z))
            A: A matrices from dynamic network (batch_size, T, dim_z, dim_z) (A_1, A_2, ..., A_T)
            B: B matrices from dynamic network (batch_size, T, dim_z, dim_u) (B_1, B_2, ..., B_T)
            C: C matrices from dynamic network (batch_size, T, dim_y, dim_z) (C_0, C_1, ..., C_(T-1)) OR (C_1, C_2, ..., C_T)
            a: Compressed observations (batch_size, T, dim_a) e.g. [a_0, a_1, ..., a_(T-1)]
            u: Control inputs (batch_size, T, dim_u) e.g. [u_1, u_2, ..., u_T]
        Returns: 
            log_prob_trans, log_prob_emiss, entropy: log probabilities (batch_size, T)
        """
        mu_smooth, Sigma_smooth, A, B, C, a, u, _, _ = backward_states
        batch_size = A.shape[0]
        T = A.shape[1]

        I = torch.eye(self.dim_a, requires_grad=False, device=self.device)
        if R is None:
            R = self.R.repeat(batch_size, T, 1, 1)
        else:
            R = R
        R = s * R + self.eps * I

        Q_batch = self.Q.repeat(batch_size, T, 1, 1)

        mu_smooth = torch.squeeze(mu_smooth)
        mvn_smooth = tdist.MultivariateNormal(mu_smooth, covariance_matrix=Sigma_smooth)

        # from t=0 to T-1
        z_smooth = mvn_smooth.rsample() # (batch_size, T, dim_z)
        
        # entropy \prod_{t=0}^{T-1} p(z_t|y_{0:T-1}, u_{1:T-1})
        entropy = mvn_smooth.log_prob(z_smooth) # (batch_size, T)

        # distribution of the initial state p(z_0)
        mu_0 = self.mu_0.repeat(batch_size, 1, 1).squeeze(-1)
        Sigma_0 = self.Sigma_0.repeat(batch_size, 1, 1)
        mvn_0 = tdist.MultivariateNormal(mu_0, covariance_matrix=Sigma_0) # ((batch_size, dim_z), (batch_size, dim_z, dim_z))
        log_prob_0 = mvn_0.log_prob(z_smooth[:, 0, :]).unsqueeze(-1) # (batch_size, 1)
            
        # re-use original transitions and emission
        A = A[:, :-1, :, :] # (batch_size, T-1, dim_z, dim_z)
        B = B[:, :-1, :, :]
        u = u[:, :-1, :].unsqueeze(-1) # (batch_size, T-1, dim_u, 1)

        z_smooth_trans = z_smooth[:, :-1, :].unsqueeze(-1) # (batch_size, T-1, dim_z, 1)
        z_smooth_emiss = z_smooth.unsqueeze(-1) # (batch_size, T, dim_z, 1)

        # transition distribution \prod_{t=1}^{T-1} p(z_t|z_{t-1}, u_{t})
        A = A.reshape(A.shape[0] * A.shape[1], *A.shape[2:]) # (batch_size, T-1, dim_z, dim_z) --> # (batch_size * T-1, dim_z, dim_z)
        z_smooth_trans = z_smooth_trans.reshape(z_smooth_trans.shape[0] * z_smooth_trans.shape[1], 
                                                *z_smooth_trans.shape[2:]) # (batch_size, T-1, dim_z, 1) --> (batch_size * T-1, dim_z, 1)
        Az_tm1 = torch.bmm(A, z_smooth_trans) 
        Az_tm1 = Az_tm1.reshape(batch_size, T-1, self.dim_z, 1)

        u = u.reshape(u.shape[0] * u.shape[1], *u.shape[2:]) # (batch_size, T-1, dim_z, 1) --> (batch_size * T-1, dim_z, 1)
        B = B.reshape(B.shape[0] * B.shape[1], *B.shape[2:])
        Bu_t = torch.bmm(B, u)
        Bu_t = Bu_t.reshape(batch_size, T-1, self.dim_z, 1)
        u = u.reshape(batch_size, T-1, self.dim_u, 1) 

        mu_trans = Az_tm1 + Bu_t
        mu_trans = torch.squeeze(mu_trans) # (batch_size, T-1, dim_z)
        mvn_trans = tdist.MultivariateNormal(mu_trans, covariance_matrix=Q_batch[:, :-1, :, :]) # ((batch_size, T-1, dim_z), (batch_size, T-1, dim_z, dim_z))

        log_prob_trans = mvn_trans.log_prob(z_smooth[:, 1:, :]) # (batch_size, T-1)
        log_prob_trans = torch.cat((log_prob_0, log_prob_trans), dim=1)

        # emission distribution \prod_{t=0}^{T-1} p(a_t|z_t)
        C = C.reshape(C.shape[0] * C.shape[1], *C.shape[2:])
        z_smooth_emiss = z_smooth_emiss.reshape(z_smooth_emiss.shape[0] * z_smooth_emiss.shape[1], 
                                                *z_smooth_emiss.shape[2:])
        Cz_t = torch.bmm(C, z_smooth_emiss)
        Cz_t = Cz_t.reshape(batch_size, T, self.dim_a, 1) # (batch_size, T, dim_a, 1)

        mu_emiss = torch.squeeze(Cz_t) # (batch_size, T, dim_a)
        mvn_emiss = tdist.MultivariateNormal(mu_emiss, covariance_matrix=R) # ((batch, T, dim_a), (batch_size, T, dim_a, dim_a))
        log_prob_emiss = mvn_emiss.log_prob(a) # (batch_size, T)

        return (log_prob_trans + log_prob_emiss - entropy)