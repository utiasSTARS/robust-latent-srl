"""
References:
https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py
"""
import torch.utils.data as data
import torch

import os
import sys
import pickle
import numpy as np

from srl.srl.transforms import GaussianNoise, Obstruct, Dropped

def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.
    Args:
        filename (string): path to a file
        extensions (iterable of strings): extensions to consider (lowercase)
    Returns:
        bool: True if the filename ends with one of given extensions
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)

def find_subdir(dir):
    """
    Finds the subdirectories in a directory.

    Args:
        dir (string): Root directory path.
    Returns:
        subdirs (list): where subdirs are relative to (dir).
    """
    if sys.version_info >= (3, 5):
        # Faster and available in Python 3.5 and above
        subdirs = [d.name for d in os.scandir(dir) if d.is_dir()]
    else:
        subdirs = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    subdirs.sort()
    return subdirs

def make_dataset_traj(dir, extensions):
    """Generate a list of data file paths."""
    data_path_list = []
    dir = os.path.expanduser(dir)
    for batch in find_subdir(dir):
            batch_path = os.path.join(dir, batch)
            for root, _, fnames in sorted(os.walk(batch_path)):
                for fname in sorted(fnames):
                    if has_file_allowed_extension(fname, extensions):
                        path = os.path.join(root, fname)
                        data_path_list.append(path)
    return data_path_list

def pickle_loader(path):
    """A data loader for pickle files."""
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

class DatasetUnsupervisedCached(data.Dataset):
    """Unsupervised dataset with no labels from a single cache file.
    """
    def __init__(self, dir, loader=pickle_loader, transform=None,
                render_h=64, render_w=64, p_noise=0, p_obstruction=0, p_dropped=0):
        """
        Args:
            dir (string): Directory of the cache.
            loader (callable): Function to load a sample given its path.
        """
        self.dir = dir
        self.p_obstruction = p_obstruction
        self.obstruct = Obstruct(p=p_obstruction)
        self.p_dropped = p_dropped
        self.drop = Dropped(p=p_dropped)
        self.p_noise = p_noise
        self.add_noise = GaussianNoise(p=p_noise, std=0.75, mean=0)
        self.transform = transform

        print("Loading cache for dataset")
        data = loader(dir) 

        if len(data) == 2:
            cached_data_raw, self.cached_data_actions = data
        elif len(data) == 3:
            cached_data_raw, self.cached_data_actions, self.cached_data_state = data
        else:
            raise NotImplementedError()

        print("Formating dataset")
        og_shape = cached_data_raw.shape
        cached_data_raw = cached_data_raw.reshape(og_shape[0], og_shape[1], render_h, render_w, 3)

        self.cached_data = torch.zeros(og_shape[0], og_shape[1], 1, render_h, render_w)
        for ii in range(og_shape[0]):
            for tt in range(og_shape[1]):
                self.cached_data[ii, tt, 0, :, :] = transform(cached_data_raw[ii, tt, :, :, :])
            
    def __len__(self):
        return self.cached_data.shape[0]

    def __getitem__(self, idx):
        """
        Args:
            index (int): Index

        Returns:
            sample (dict): 
        """
        assert(idx < self.__len__()), "Index must be lower than dataset size " + str(self.__len__())
        img = self.cached_data[idx] # (T, 1, res, res) or (T, 2, res, res)
        a = self.cached_data_actions[idx] # (T, 1)

        # Randomly add noise in training dataset
        if self.p_noise > 0: 
            for tt in range(img.shape[0]):
                img[tt, 0, :, :] = self.add_noise(img[tt, 0, :, :])

        # Randomly drop or obstruct whole measurements in training dataset
        if self.p_dropped > 0: 
            for tt in range(img.shape[0]):
                img[tt, 0, :, :] = self.drop(img[tt, 0, :, :])

        # Randomly obstruct images in training dataset
        if self.p_obstruction > 0: 
            for tt in range(img.shape[0]):
                img[tt, 0, :, :] = self.obstruct(img[tt, 0, :, :])

        sample = {'images':img, # (T, 1, res, res) or (T, 2, res, res)
                  'actions': a}

        return sample

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Dir Location: {}\n'.format(self.dir)
        tmp = '    Image Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


class DatasetRealLifeCache(data.Dataset):
    """ A dataset that treats each trajectory as a data point.
    """
    def __init__(self, cached_data_path, transform=lambda x: x, loader=pickle_loader):
        assert os.path.exists(cached_data_path)
        self.cached_data_path = cached_data_path
        self.transform = transform

        print("Loading cache for dataset")
        self.data = loader(cached_data_path)

        drop_p = 1e-2
        obstruct_p = 1e-2
        noise_p = 1e-2
        self.random_t = [Dropped(p=drop_p),
                         Obstruct(p=obstruct_p, value=0),
                         GaussianNoise(p=noise_p, std=0.25, mean=0.25)]

    def __len__(self):
        return len(self.data["trajectories"])

    def __getitem__(self, idx):
        assert(idx < self.__len__()), "Index must be lower than dataset size " + str(self.__len__())

        (observations, actions, rewards, dones) = self.data["trajectories"][idx]

        traj_obs = torch.tensor(self.transform(observations[1:]))
        for timestep, traj_ob in enumerate(traj_obs):
            t = np.random.choice(self.random_t)
            traj_obs[timestep, 0] = t(traj_ob[0])

        return {"images": traj_obs,
                "actions": actions,
                "rewards": rewards,
                "dones": dones}
