import cv2
import logging
import numpy as np
import gym
import time
from gym import spaces
from multiprocessing import Array, Value

from senseact.devices.real_sense.real_sense_communicator import RealSenseCommunicator
from senseact.devices.ur import ur_utils
from senseact.devices.ur.ur_setups import setups
from senseact.rtrl_base_env import RTRLBaseEnv
from senseact.sharedbuffer import SharedBuffer
from senseact import utils


class RealSenseEnv(RTRLBaseEnv, gym.core.Env):
    def __init__(self,
                 camera_res=(3, 480, 640),
                 time_limit=10,
                 hosts=('localhost',),
                 ports=(5000,),
                 rng=np.random,
                 **kwargs):
        assert time_limit > 0
        assert len(hosts) == len(ports)

        self.num_cameras = len(hosts)
        self.buffer_len = 2
        self.action_dim = 2
        self.camera_dim = int(np.product(camera_res))
        self.input_dim = self.num_cameras * self.camera_dim
        self.camera_res = camera_res

        self._time_limit = time_limit

        self.rng = rng

        self._action_space = spaces.Discrete(2)
        self._observation_space = spaces.Box(
            low=0, high=1, shape=(self.input_dim,), dtype=np.float32)

        # Setup communicator and buffer
        communicator_setups = {}
        self._camera_images_ = {}

        for idx, (host, port) in enumerate(zip(hosts, ports)):
            communicator_setups[f'Camera_{idx}'] = {
                'Communicator': RealSenseCommunicator,
                # have to read in this number of packets everytime to support
                # all operations
                'num_sensor_packets': self.buffer_len,
                'kwargs': {
                    'host': host,
                    'port': port,
                    'num_channels': camera_res[0],
                    'height': camera_res[1],
                    'width': camera_res[2]
                }
            }

            self._camera_images_[f'Camera_{idx}'] = np.frombuffer(
                Array('f', self.camera_dim).get_obj(), dtype='float32')

        super(RealSenseEnv, self).__init__(
            communicator_setups=communicator_setups,
            action_dim=self.action_dim,
            observation_dim=self.input_dim,
            **kwargs
        )

        self._obs_ = np.zeros(shape=self.input_dim)
        self.episode_steps = Value('i', 0)

    def _reset_(self):
        self.done = False
        self.episode_steps.value = 0
        self._sensor_to_sensation_()

    def _compute_sensation_(self, name, sensor_window, timestamp_window, index_window):
        if name.startswith('Camera'):
            image = np.array(sensor_window[-1])
            camera_idx = int(name.split("_")[1])
            np.copyto(self._camera_images_[name], image.flatten())
            np.copyto(self._obs_[camera_idx * self.camera_dim:(camera_idx + 1) * self.camera_dim], image.flatten())
    
        reward = self._compute_reward()
        
        return np.concatenate((self._obs_, [reward], [self.done]))

    def _compute_actuation_(self, action, timestamp, index):
        if action[1]:
            self.done = True

    def _check_done(self, env_done):
        self.episode_steps.value += 1
        return env_done or (self._time_limit < self.episode_steps.value)

    def _compute_reward(self):
        return self.rng.normal(loc=0, scale=self.episode_steps.value)

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    def render(self):
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        images = []
        for idx in range(self.num_cameras):
            images.append(self._camera_images_[f'Camera_{idx}'].reshape(self.camera_res).transpose(1, 2, 0))
        images = np.hstack(images)
        cv2.imshow('RealSense', images)
        cv2.waitKey(1)

    def terminate(self):
        """Gracefully terminates environment processes."""
        super(RealSenseEnv, self).close()


if __name__ == "__main__":
    hosts = ('localhost',)
    ports = (5000,)
    env = RealSenseEnv(
        time_limit=10,
        hosts=hosts,
        ports=ports)
    env.start()
    
    for episode in range(10):
        print(f"Episode: {episode}")
        done = False
        obs = env.reset()
        while not done:
            env.render()
            obs, reward, done, _ = env.step([1, 0])

    env.close()
