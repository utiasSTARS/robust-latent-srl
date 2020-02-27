import cv2
import numpy as np
import time
import gym
import sys
from multiprocessing import Array, Value

from senseact.rtrl_base_env import RTRLBaseEnv
from senseact.envs.ur.reacher_env import ReacherEnv
from senseact.devices.real_sense.real_sense_communicator import RealSenseCommunicator
from senseact.devices.ur import ur_utils
from senseact.devices.ur.ur_setups import setups
from senseact.sharedbuffer import SharedBuffer
from senseact import utils


class ReacherEnvWithRealSense(ReacherEnv, gym.core.Env):
    def __init__(self,
                 setup,
                 camera_hosts=('localhost',),
                 camera_ports=(5000,),
                 camera_res=(3, 480, 640),
                 host=None,
                 q_start_queue=None,
                 q_target=None,
                 dof=6,
                 control_type='position',
                 derivative_type='none',
                 target_type='position',
                 reset_type='random',
                 reward_type='linear',
                 deriv_action_max=10,
                 first_deriv_max=10,  # used only with second derivative control
                 vel_penalty=0,
                 obs_history=1,
                 actuation_sync_period=1,
                 episode_length_time=None,
                 episode_length_step=None,
                 rllab_box = False,
                 servoj_t=ur_utils.COMMANDS['SERVOJ']['default']['t'],
                 servoj_gain=ur_utils.COMMANDS['SERVOJ']['default']['gain'],
                 speedj_a=ur_utils.COMMANDS['SPEEDJ']['default']['a'],
                 speedj_t_min=ur_utils.COMMANDS['SPEEDJ']['default']['t_min'],
                 movej_t=2, # used for resetting
                 accel_max=None,
                 speed_max=None,
                 dt=0.008,
                 delay=0.0,  # to simulate extra delay in the system
                 **kwargs):
        assert len(camera_hosts) == len(camera_ports)

        self.num_cameras = len(camera_hosts)
        self.camera_res = camera_res
        self.camera_dim = int(np.product(camera_res))
        self.buffer_len = obs_history
        self.q_start_queue = q_start_queue
        self.q_target = q_target
        
        # Setup camera communicators and buffer
        communicator_setups = {}
        self._camera_images_ = {}

        for idx, (camera_host, camera_port) in enumerate(zip(camera_hosts, camera_ports)):
            communicator_setups['Camera_{}'.format(idx)] = {
                'Communicator': RealSenseCommunicator,
                # have to read in this number of packets everytime to support
                # all operations
                'num_sensor_packets': self.buffer_len,
                'kwargs': {
                    'host': camera_host,
                    'port': camera_port,
                    'num_channels': camera_res[0],
                    'height': camera_res[1],
                    'width': camera_res[2]
                }
            }

            self._camera_images_['Camera_{}'.format(idx)] = np.frombuffer(
                Array('f', self.camera_dim).get_obj(), dtype='float32')

        print("Setup Reacher Environment")
        # Setup UR environment
        super(ReacherEnvWithRealSense, self).__init__(
            setup,
            host=host,
            dof=dof,
            control_type=control_type,
            derivative_type=derivative_type,
            target_type=target_type,
            reset_type=reset_type,
            reward_type=reward_type,
            deriv_action_max=deriv_action_max,
            first_deriv_max=first_deriv_max,  # used only with second derivative control
            vel_penalty=vel_penalty,
            obs_history=obs_history,
            actuation_sync_period=actuation_sync_period,
            episode_length_time=episode_length_time,
            episode_length_step=episode_length_step,
            rllab_box = rllab_box,
            servoj_t=servoj_t,
            servoj_gain=servoj_gain,
            speedj_a=speedj_a,
            speedj_t_min=speedj_t_min,
            movej_t=movej_t, # used for resetting
            accel_max=accel_max,
            speed_max=speed_max,
            dt=dt,
            delay=delay,  # to simulate extra delay in the system
            communicator_setups=communicator_setups,
            **kwargs)

        # Update the observation space from ReacherEnv to include camera
        self.joint_dim = int(np.product(self._observation_space.shape))
        self.input_dim = self.joint_dim + self.num_cameras * self.camera_dim

        if rllab_box:
            from rllab.spaces import Box as RlBox  # use this for rllab TRPO
            Box = RlBox
        else:
            from gym.spaces import Box as GymBox  # use this for baselines algos
            Box = GymBox
            
        self._observation_space = Box(
            low=-np.concatenate(
                (np.zeros(self.num_cameras * self.camera_dim), self._observation_space.low)),
            high=np.concatenate(
                (np.ones(self.num_cameras * self.camera_dim), self._observation_space.high))
        )

        print("Communicators Setup")
        RTRLBaseEnv.__init__(self, communicator_setups=communicator_setups,
                            action_dim=len(self.action_space.low),
                            observation_dim=len(self.observation_space.low),
                            dt=dt,
                            **kwargs)

        self._obs_ = np.zeros(self.input_dim)

    def _reset_(self):
        print ("Reset Arm")
        q_start = None
        if self.q_start_queue:
            q_start = np.array(self.q_start_queue.pop(0))
        
        resetted = False
        while not resetted:
            super(ReacherEnvWithRealSense, self)._reset_(q_start=q_start,
                                                         q_target=self.q_target)
            self._sensor_to_sensation_()
            sensation, _, _ = self._sensation_buffer.read()
            ur5_obs = sensation[0][:-2]
            print(ur5_obs[-8:][:2], q_start)
            if q_start is None or np.allclose(ur5_obs[-8:][:2], q_start, atol=1e-2, rtol=1e-2):
                resetted = True
        print("Reset Completed")

    def _compute_sensation_(self, name, sensor_window, timestamp_window, index_window):
        if name == 'UR5':
            ur5_obs = super(ReacherEnvWithRealSense, self)._compute_sensation_(name, sensor_window, timestamp_window, index_window)
            np.copyto(self._obs_[-self.joint_dim:], ur5_obs[:-2])
        elif name.startswith('Camera'):
            image = np.array(sensor_window[-1])
            camera_idx = int(name.split("_")[1])
            np.copyto(self._camera_images_[name], image.flatten())
            np.copyto(self._obs_[camera_idx * self.camera_dim:(camera_idx + 1) * self.camera_dim], image.flatten())

        return np.concatenate((self._obs_, [self._reward_.value], [0]))

    def render(self):
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        images = []
        for idx in range(self.num_cameras):
            images.append(self._camera_images_['Camera_{}'.format(idx)].reshape(self.camera_res).transpose(1, 2, 0))
        images = np.hstack(images)
        cv2.imshow('RealSense', images)
        cv2.waitKey(1)
