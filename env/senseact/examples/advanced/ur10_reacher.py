import time
import copy
import numpy as np

from multiprocessing import Process, Value, Manager

from senseact.envs.ur.reacher_env import ReacherEnv
from senseact.envs.real_sense.reacher_env_with_real_sense import ReacherEnvWithRealSense
from senseact.utils import tf_set_seeds, NormalizedEnv
from helper import create_callback


def main():
    # use fixed random state
    rand_state = np.random.RandomState(1).get_state()
    np.random.set_state(rand_state)

    # Create UR10 Reacher2D environment
    env = ReacherEnvWithRealSense(
            setup="UR10_default",
            camera_hosts=('localhost',),
            camera_ports=(5000,),
            camera_res=(3, 480, 640),
            host=None,
            dof=2,
            control_type="velocity",
            target_type="position",
            reset_type="zero",
            reward_type="precision",
            derivative_type="none",
            deriv_action_max=5,
            first_deriv_max=2,
            accel_max=1.4,
            speed_max=1.0,
            speedj_a=1.4,
            episode_length_time=4.0,
            episode_length_step=None,
            actuation_sync_period=1,
            dt=0.5,
            run_mode="multiprocess",
            rllab_box=False,
            movej_t=2.0,
            delay=0.0,
            random_state=rand_state
        )
    env = NormalizedEnv(env)
    # Start environment processes
    env.start()


    # Create and start plotting process
    plot_running = Value('i', 1)
    shared_returns = Manager().dict({"write_lock": False,
                                     "episodic_returns": [],
                                     "episodic_lengths": [], })
    # Spawn plotting process
    pp = Process(target=plot_ur10_reacher, args=(env, 2048, shared_returns, plot_running))
    pp.start()

    # Create callback function for logging data from baselines TRPO learn
    kindred_callback = create_callback(shared_returns)

    # env.action_space.shape
    # Train baselines TRPO
    for episode in range(10):
        print(f"Episode: {episode + 1}")
        done = False
        timestep = 0
        curr_obs = env.reset()
        while not done:
            if timestep % 3 == 0:
                action = np.random.normal(scale=0.1, size=(2,))
            print(action)
            next_obs, reward, done, _ = env.step(action)

            timestep += 1
            curr_obs = next_obs

            if timestep == 15:
                done = True

    # Safely terminate plotter process
    plot_running.value = 0  # shutdown ploting process
    time.sleep(2)
    pp.join()

    env.close()


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
    ax3.set_xlabel("Z", fontsize=14)
    h = ax3.set_ylabel("Y", fontsize=14)
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
        ax1.set_title("X-Y plane", fontsize=14)
        ax1.set_xlim(ax1.get_xlim()[::-1])
        ax1.set_ylim(ax1.get_ylim()[::-1])

        hl3.set_ydata([env._x_target_[1]])
        hl3.set_xdata([env._x_target_[0]])
        hl4.set_ydata([env._x_[1]])
        hl4.set_xdata([env._x_[0]])
        ax3.set_ylim([env._end_effector_low[1], env._end_effector_high[1]])
        ax3.set_xlim([env._end_effector_low[0], env._end_effector_high[0]])
        ax3.set_title("Y-Z plane", fontsize=14)
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


if __name__ == '__main__':
    main()
