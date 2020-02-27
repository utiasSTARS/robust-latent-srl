"""Collect dataset specifically for the pendulum env."""
import sys
from classic_control_pixel.pendulum import Pendulum
import numpy as np
import matplotlib.pyplot as plt
import _pickle as pkl
from torchvision import transforms

TRAJECTORIES = 2048
T = 16
REPEAT = 1
SAVE_PATH = "/path/to/save/"
RES = 16
INCLUDE_ANGLE = True

def angle_normalize(x):
    return (((x+np.pi) % (2*np.pi)) - np.pi)

def main():
    env = Pendulum(render_h=RES, render_w=RES)
    obs = env.reset()

    cached_data_imgs = np.empty((TRAJECTORIES, 
                                    T*REPEAT,
                                    3 * RES * RES
                                    ), dtype=np.uint8)
    cached_data_actions = np.empty((TRAJECTORIES, 
                                    T*REPEAT,
                                    1), dtype=np.float32)
    cached_data_angles = np.empty((TRAJECTORIES, 
                                    T*REPEAT,
                                    2), dtype=np.float32)

    n_bytes = cached_data_imgs.size * cached_data_imgs.itemsize + \
                cached_data_actions.size * cached_data_actions.itemsize + \
                cached_data_angles.size * cached_data_angles.itemsize
                
    print("Saving {} GBs of data in {}".format(n_bytes/1e9, SAVE_PATH))

    for traj in np.arange(TRAJECTORIES):
        th = np.random.uniform(0, np.pi * 2)
        thdot = np.random.uniform(-1,1)
        state = np.array([th, thdot])
        obs = env.reset(state=state)
        count = 0
        for _ in np.arange(T):
            u = np.random.normal(0, 1)
            # u = np.random.uniform(-2, 2, size=(1,))
            u = np.clip(u, -2, 2)
            for _ in np.arange(REPEAT):
                obs, _, _, _ = env.step(u=np.array([u]))
                cached_data_imgs[traj, count, :] = obs.flatten()
                cached_data_actions[traj, count, :] = u
                if INCLUDE_ANGLE:
                    cached_data_angles[traj, count, :] = np.array([angle_normalize(env.state[0]),
                                                                    env.state[1]])
                count+=1

    if INCLUDE_ANGLE:
        cached_data = (cached_data_imgs, cached_data_actions, cached_data_angles)
    else:
        cached_data = (cached_data_imgs, cached_data_actions)

    pickle_out = open(SAVE_PATH + "pendulum{}_total_{}_traj_{}_repeat_{}_with_angle_train.pkl".format(RES,TRAJECTORIES, T, REPEAT), "wb")
    pkl.dump(cached_data, pickle_out, protocol=4)
    pickle_out.close()
    
    env.close()

if __name__ == "__main__":
    main()
