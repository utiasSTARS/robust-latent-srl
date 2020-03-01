import _pickle as pickle
import argparse
import copy
import json
import numpy as np
import operator
import os
import time
import torch
import torchvision.transforms as torch_transforms

from argparse import Namespace
from functools import partial
from multiprocessing import Process, Value, Manager

from args.parser import parse_control_experiment_args
from data_collection.storage import Storage
from experiments.mpc import *
from senseact.envs.real_sense.reacher_env_with_real_sense import ReacherEnvWithRealSense
from srl.srl.networks import (FullyConvEncoderVAE,
                              FullyConvDecoderVAE,
                              LGSSM,
                              RNNAlpha)
from srl.srl.transforms import (AsType,
                                DownSample,
                                Dropped,
                                GaussianNoise,
                                NormalizeImage,
                                Obstruct,
                                Reshape,
                                Transpose)
from srl.srl.utils import load_models

def get_alpha_scaling_torch(l_train_avg, x, x_rec, dim):
    diff = (x - x_rec)**2
    L_rec = torch.sum(diff, dim=(1,2,3)) / (diff.shape[-1] * diff.shape[-2])
    alpha = torch.log(1 + (L_rec/l_train_avg)).unsqueeze(-1)
    alpha = torch.diag_embed(alpha.repeat(1, dim))
    alpha = 10 * alpha.reshape(1, x.shape[0], *alpha.shape[1:])
    return alpha

def ur10_control(args):
    assert len(args.camera_res) == 3
    assert len(args.hosts) == len(args.ports) > 0
    assert args.dt > 0
    assert args.repeat_actions >= 0
    assert args.timeout > 0
    assert args.speed_max > 0
    assert os.path.isdir(args.model_path)

    num_feature_dim = 8
    L_TRAIN_AVG = 0.000939284324645996 # reacher
    ood_detection = partial(get_alpha_scaling_torch, l_train_avg=L_TRAIN_AVG)

    with open("{}".format(args.args_output_file), "wb") as f:
        pickle.dump(args, f)

    # use fixed random state
    rand_state = np.random.RandomState(args.seed).get_state()
    np.random.set_state(rand_state)

    storage = Storage(args.dbname)

    # Load model
    print("Loading Model")
    with open(os.path.join(args.model_path, "hyperparameters.txt"), "r") as f:
        model_args = Namespace(**json.load(f))

    enc, dec, lgssm = load_models(args.model_path, model_args, device=args.device)

    goal_data = np.load(args.goal_path)
    goal_gt = goal_data[-num_feature_dim:]
    goal_image = goal_data[:-num_feature_dim].reshape(model_args.dim_x[1:])

    # TARGET [-1.49640781 -1.75541813]
    # q_start_queue = [#[-1.6, -1.3],
    #                 #  [-2.0, -1.4],
    #                 #  [-2.2, -1.6],
    #                 #  [-2.35, -1.2],
    #                 #  [-0.8, -1.85],
    #                  [-1.1, -1.7],##
    #                  [-2.5, -0.66],##E STOP
    #                  [-2.2, -0.6],##
    #                  [-2.68, -0.07],##
    #                  [-2.69, 0.28]]

    q_start_queue = [[-1.50518352, -1.80493862],
                    [-1.16035873, -1.89097912],
                    [-1.1, -1.7],##
                    [-2.5, -0.66],##E STOP
                    [-2.2, -0.6],##
                    [-2.68, -0.07],##
                    [-2.21267349, -0.69658262],
                    [-0.5753792,  -1.88098842],
                    [-1.41478187, -1.4714458]]
    # q_start_queue = [[-1.41478187, -1.4714458], [-0.5753792,  -1.88098842]]

    q_target = goal_gt[:2]
    print("TARGET {}".format(q_target))

    # Create UR10 Reacher2D environment
    print("Creating Environment")
    env = ReacherEnvWithRealSense(
            setup="UR10_default",
            camera_hosts=args.hosts,
            camera_ports=args.ports,
            camera_res=args.camera_res,
            host=None,
            q_start_queue=q_start_queue,
            q_target=q_target,
            dof=2,
            control_type="velocity",
            target_type="position",
            reset_type="random",
            reward_type="precision",
            derivative_type="none",
            deriv_action_max=5,
            first_deriv_max=2,
            accel_max=1.4,
            speed_max=args.speed_max,
            speedj_a=1.4,
            episode_length_time=None,
            episode_length_step=args.timeout,
            actuation_sync_period=1,
            dt=args.dt,
            run_mode="multiprocess",
            rllab_box=False,
            movej_t=2.0,
            delay=0.0,
            random_state=rand_state
        )

    # Create and start plotting process
    plot_running = Value('i', 1)
    shared_returns = Manager().dict({"write_lock": False,
                                     "episodic_returns": [],
                                     "episodic_lengths": [], })
    # Spawn plotting process
    print("Spawning plotting process")
    pp = Process(target=plot_ur10_reacher, args=(env, 2048, shared_returns, plot_running))
    pp.start()

    render = lambda: None
    if args.render and env.render:
        render = env.render

    print ("Setup Goal and transforms")
    # Setup transform
    obs_transform = torch_transforms.Compose(transforms=[Reshape(args.camera_res),
                                                   NormalizeImage(const=1/255),
                                                   AsType(dtype=np.uint8),
                                                   Transpose(transpose_shape=(1, 2, 0)),
                                                   DownSample(*model_args.dim_x[1:]),
                                                   torch_transforms.ToPILImage(),
                                                   torch_transforms.Grayscale(num_output_channels=1),
                                                   torch_transforms.ToTensor()])

    # Randomly generate goal image
    goal_img = torch.tensor(goal_image.reshape(model_args.dim_x), dtype=torch.float, device=args.device)
    x_goal = goal_img.unsqueeze(0).expand(args.T, *model_args.dim_x).reshape(args.T, *model_args.dim_x)
    u_goal = torch.zeros(1, args.T, model_args.dim_u, device=args.device)

    a_goal, a_goal_mu, a_goal_logvar = enc(x_goal)
    a_goal_cov = torch.exp(a_goal_logvar)
    R_goal_cov = None
    s_goal = torch.tensor(1.0, requires_grad=False, device=args.device)
    a_goal = a_goal.reshape(1, args.T, model_args.dim_a)
    with torch.no_grad():
        mu_goal, Sigma_goal, alpha_goal, h_goal = lgssm.initialize(a_goal, u_goal, R=R_goal_cov)
    z_goal = mu_goal[0, :, 0].cpu().detach().numpy()

    # Setup MPC
    umin = -0.5 * np.ones((model_args.dim_u))
    umax = 0.5 * np.ones((model_args.dim_u))
    zmin = -np.inf * np.ones((model_args.dim_z))
    zmax = np.inf * np.ones((model_args.dim_z))
    mpc_Q = 1.0 * sparse.eye(model_args.dim_z)
    mpc_R = 1.0 * sparse.eye(model_args.dim_u)

    mpc = MPC(model_args.dim_z, model_args.dim_u, args.mpc_horizon, mpc_Q, mpc_R, zmin, zmax, umin, umax)

    R_update = torch.eye(model_args.dim_a, requires_grad=False, device=args.device).unsqueeze(0) * model_args.emission_noise

    t_dropped = Dropped(p=1.0)
    t_obstruct = Obstruct(p=1.0, value=1)
    t_noise = GaussianNoise(p=1.0, std=0.25, mean=0.25)
    t_image_transform = torch_transforms.ColorJitter()

    # =========== SETUP OOD
    ood_transform = t_dropped
    clean_frequency = 2
    dirty_frequency = 2

    # GT is clean first. LE is dirty first
    apply_ood = operator.le
    # apply_ood = operator.gt

    def transform_initialization(curr_x):
        curr_x[0] = ood_transform(curr_x[0])
        curr_x[1] = ood_transform(curr_x[1])
        curr_x[3] = ood_transform(curr_x[3])
        return curr_x

    def transform_image(curr_x, timestep):
        if apply_ood(timestep % (clean_frequency + dirty_frequency), dirty_frequency - 1):
            curr_x = ood_transform(curr_x)
        return curr_x


    try:
        print("START CONTROL EXPERIMENT")
        storage.start()
        env.start()

        with torch.no_grad():
            for episode in range(args.num_episodes):
            # for episode in range(len(q_start_queue)):
                print("Episode: {}".format(episode + 1))
                done = False
                timestep = 0
                curr_obs = env.reset()

                # Initialize LGSSM using first observation
                # NOTE: Assume constant setting from LGSSM
                tic = time.time()
                curr_x = torch.tensor(obs_transform(curr_obs[:-num_feature_dim]),
                                    device=args.device)
                print("STARTING STATE: {}".format(curr_obs[-num_feature_dim:][:2]))

                curr_x = curr_x.expand(args.T, *model_args.dim_x).reshape(args.T, *model_args.dim_x)
                curr_x = transform_initialization(curr_x)

                curr_a, _, _ = enc(curr_x)
                curr_x_hat = dec(curr_a)

                # OOD
                ood_factor = 1.
                curr_R_cov = None
                if args.enable_ood:
                    ood_factor = ood_detection(x=curr_x, x_rec=curr_x_hat, dim=model_args.dim_a)
                    curr_R_cov = ood_factor * model_args.emission_noise
                curr_s = torch.tensor(1., requires_grad=False, device=args.device)
                curr_a = curr_a.reshape(1, args.T, model_args.dim_a)
                curr_u = torch.zeros(1, args.T, model_args.dim_u, device=args.device)
                mu, Sigma, alpha, h = lgssm.initialize(curr_a, curr_u, s=curr_s, R=curr_R_cov)
                curr_z = mu[0, :, 0].cpu().detach().numpy()

                mpc_curr_u = torch.zeros((1, args.mpc_horizon * args.repeat_actions, model_args.dim_u)).float().to(args.device)
                toc = time.time()
                # print("TIME TAKEN FOR INITIALIZATION: {}".format(toc - tic))

                while not done:
                    # print("TIMESTEP: {} ========================".format(timestep))

                    if timestep % args.repeat_actions == 0:
                        tic = time.time()
                        _, _, _, _, A, B, _ = lgssm.predict(mu_tn1=mu,
                                                            Sigma_tn1=Sigma,
                                                            alpha_t=alpha,
                                                            h_t=h,
                                                            u_f=mpc_curr_u)
                        toc = time.time()
                        # print("TIME TAKEN FOR LGSSM PREDICT: {}".format(toc - tic))

                        A = A[0].reshape(args.mpc_horizon, args.repeat_actions, model_args.dim_z, model_args.dim_z)
                        B = B[0].reshape(args.mpc_horizon, args.repeat_actions, model_args.dim_z, model_args.dim_u)

                        # Approiximate transformations for A and B
                        mpc_A = A[:, -1]
                        mpc_B = B[:, -1]
                        for action_i in range(args.repeat_actions - 2, -1, -1):
                            mpc_B = mpc_B + torch.bmm(mpc_A, B[:, action_i])
                            mpc_A = torch.bmm(mpc_A, A[:, action_i])

                        mpc_A = mpc_A.cpu().detach().numpy()
                        mpc_B = mpc_B.cpu().detach().numpy()

                        # Compute actions using MPC
                        tic = time.time()
                        _ = mpc.run_mpc(mpc_A, mpc_B, curr_z, z_goal)
                        toc = time.time()
                        # print("TIME TAKEN FOR MPC CALL: {}".format(toc - tic))

                        mpc_curr_u = mpc.u.value
                        action = np.copy(mpc_curr_u[:, 0])
                        # print("ACTION FROM MPC: {}".format(action))

                        # Repeat actions
                        mpc_curr_u[:, :-1] = mpc_curr_u[:, 1:]
                        mpc_curr_u[:, -1] = 0 
                        mpc_curr_u = torch.tensor(mpc_curr_u, device=args.device).float().repeat_interleave(args.repeat_actions, 1).transpose(1, 0).unsqueeze(0)

                    render()
                    # dummy_action = np.zeros(model_args.dim_u)
                    # print(action)
                    # tic = time.time()
                    next_obs, reward, done, _ = env.step(action)
                    # toc = time.time()
                    # print("TIME FOR STEP: {}".format(toc - tic))
                    
                    storage.save_transition(
                        episode,
                        timestep,
                        curr_obs,
                        action,
                        reward,
                        done,
                        next_obs
                    )

                    timestep += 1
                    curr_obs = next_obs

                    tic = time.time()
                    curr_x = torch.tensor(obs_transform(curr_obs[:-num_feature_dim]),
                                          device=args.device).reshape(1, *model_args.dim_x)

                    # if timestep % 4 <= 1:
                    #     curr_x[0] = t_dropped(curr_x[0])
                    curr_x[0] = transform_image(curr_x[0], timestep)

                    curr_u = torch.from_numpy(action).unsqueeze(0).float().to(args.device) # (1, dim_u)
                    curr_a, _, _ = enc(curr_x)
                    curr_x_hat = dec(curr_a)
                    if args.enable_ood:
                        ood_factor = ood_detection(x=curr_x, x_rec=curr_x_hat, dim=model_args.dim_a).squeeze(0)
                    mu, Sigma, _, _ = \
                        lgssm.predict_update(mu, Sigma, alpha, curr_u)

                    mu, Sigma, _ = \
                        lgssm.filter_update(mu, Sigma, 
                                            alpha, curr_a, R=R_update * ood_factor)
                    curr_z = mu[0, :, 0]
                    alpha, h = lgssm.alpha_net(curr_z.reshape(1, 1, -1), h) # (1, 1, args.mpc_horizon)
                    alpha = alpha[:, 0, :]
                    curr_z = curr_z.cpu().detach().numpy() # (dim_z,)
                    toc = time.time()
                    # print("TIME TAKEN FOR UPDATING LGSSM: {}".format(toc - tic))

                    if timestep == args.timeout:
                        # print("Reached Timeout Limit {}".format(args.timeout))
                        assert done
    finally:
        storage.close()
        print(env._pstop_times_)
        env.close()
        # Safely terminate plotter process
        plot_running.value = 0  # shutdown ploting process
        time.sleep(2)
        pp.join()


def plot_ur10_reacher(env, batch_size, shared_returns, plot_running):
    """Helper process for visualize the tasks and episodic returns.

    Args:
        env: An instance of ReacherEnv
        batch_size: An int representing timesteps_per_batch provided to the PPO learn function
        shared_returns: A manager dictionary object containing `episodic returns` and `episodic lengths`
        plot_running: A multiprocessing Value object containing 0/1.
            1: Continue plotting, 0: Terminate plotting loop
    """
    print ("Started plotting routine")
    import matplotlib.pyplot as plt
    plt.ion()
    time.sleep(5.0)
    fig = plt.figure(figsize=(20, 6))
    ax1 = fig.add_subplot(131)
    hl1, = ax1.plot([], [], markersize=10, marker="o", color='r')
    hl2, = ax1.plot([], [], markersize=10, marker="o", color='b')
    ax1.set_xlabel("X", fontsize=14)
    h = ax1.set_ylabel("Y", fontsize=14)
    h.set_rotation(0)
    ax3 = fig.add_subplot(132)
    hl3, = ax3.plot([], [], markersize=10, marker="o", color='r')
    hl4, = ax3.plot([], [], markersize=10, marker="o", color='b')
    ax3.set_xlabel("X", fontsize=14)
    h = ax3.set_ylabel("Z", fontsize=14)
    h.set_rotation(0)
    ax2 = fig.add_subplot(133)
    hl11, = ax2.plot([], [])
    count = 0
    old_size = len(shared_returns['episodic_returns'])
    while plot_running.value:
        plt.suptitle("Reward: {:.2f}".format(env._reward_.value), x=0.375, fontsize=14)
        hl1.set_ydata([env._x_target_[1]])
        hl1.set_xdata([env._x_target_[2]])
        hl2.set_ydata([env._x_[1]])
        hl2.set_xdata([env._x_[2]])
        ax1.set_ylim([env._end_effector_low[1], env._end_effector_high[1]])
        ax1.set_xlim([env._end_effector_low[2], env._end_effector_high[2]])
        ax1.set_title("Y-Z plane", fontsize=14)
        ax1.set_xlim(ax1.get_xlim()[::-1])
        ax1.set_ylim(ax1.get_ylim()[::-1])

        hl3.set_ydata([env._x_target_[2]])
        hl3.set_xdata([env._x_target_[0]])
        hl4.set_ydata([env._x_[2]])
        hl4.set_xdata([env._x_[0]])
        ax3.set_ylim([env._end_effector_high[2], env._end_effector_low[2]])
        ax3.set_xlim([env._end_effector_low[0], env._end_effector_high[0]])
        ax3.set_title("X-Z plane", fontsize=14)
        ax3.set_xlim(ax3.get_xlim()[::-1])
        ax3.set_ylim(ax3.get_ylim()[::-1])

        # make a copy of the whole dict to avoid episode_returns and episodic_lengths getting desync
        # while plotting
        copied_returns = copy.deepcopy(shared_returns)
        if not copied_returns['write_lock'] and  len(copied_returns['episodic_returns']) > old_size:
            # plot learning curve
            returns = np.array(copied_returns['episodic_returns'])
            old_size = len(copied_returns['episodic_returns'])
            window_size_steps = 5000
            x_tick = 1000

            if copied_returns['episodic_lengths']:
                ep_lens = np.array(copied_returns['episodic_lengths'])
            else:
                ep_lens = batch_size * np.arange(len(returns))
            cum_episode_lengths = np.cumsum(ep_lens)

            if cum_episode_lengths[-1] >= x_tick:
                steps_show = np.arange(x_tick, cum_episode_lengths[-1] + 1, x_tick)
                rets = []

                for i in range(len(steps_show)):
                    rets_in_window = returns[(cum_episode_lengths > max(0, x_tick * (i + 1) - window_size_steps)) *
                                             (cum_episode_lengths < x_tick * (i + 1))]
                    if rets_in_window.any():
                        rets.append(np.mean(rets_in_window))

                hl11.set_xdata(np.arange(1, len(rets) + 1) * x_tick)
                ax2.set_xlim([x_tick, len(rets) * x_tick])
                hl11.set_ydata(rets)
                ax2.set_ylim([np.min(rets), np.max(rets) + 50])
        time.sleep(0.01)
        fig.canvas.draw()
        fig.canvas.flush_events()
        count += 1


if __name__ == "__main__":
    args = parse_control_experiment_args()
    ur10_control(args)
