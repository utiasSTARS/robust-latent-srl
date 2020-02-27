import _pickle as pickle
import argparse
import numpy as np
import os

from torchvision import transforms

from args.parser import parse_data_cache_args
from data_collection.storage import Storage
from srl.srl.transforms import AsType, DownSample, NormalizeImage, Reshape, Transpose


def construct_data_cache_from_sql(args):
    """ Construct data cache from sqlite3 storage.
    We assume the data is a combination of images and a feature vector with the order:
    (img_1, img_2, ..., img_N, feature_vector)
    We also assume all images have the same resolution and channels
    """
    print(args.dbname)
    assert os.path.isfile(args.dbname)
    assert args.num_scalar_features >= 0
    assert args.num_images >= 0
    assert args.num_channels >= 0
    assert len(args.camera_res) == len(args.down_sample_res) == 2

    original_image_flattened_dim = args.num_channels * int(np.product(args.camera_res))
    down_sampled_image_flattened_dim = int(np.product(args.down_sample_res))

    observation_transform = transforms.Compose(transforms=[
        Reshape((args.num_channels, *args.camera_res)),
        NormalizeImage(const=1/255),
        AsType(dtype=np.uint8),
        Transpose(transpose_shape=(1, 2, 0)),
        DownSample(*args.down_sample_res),
        transforms.ToPILImage(),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        Reshape((down_sampled_image_flattened_dim))
        ])

    storage = Storage(args.dbname)
    num_episodes = storage.get_num_episodes()
    print("{} episodes to process".format(num_episodes))

    cached_data_trajectories = []

    for episode_id in range(num_episodes):
        if (episode_id + 1) % 100 == 0:
            print("Processed {} episodes".format(episode_id + 1))

        (observations, actions, rewards, dones) = storage.get_episode(episode_id)

        num_timesteps = len(actions)
        np_observations = np.empty(
            shape=(num_timesteps + 1, args.num_images * down_sampled_image_flattened_dim + args.num_scalar_features),
            dtype=np.float
        )
        np_actions = np.empty(shape=(num_timesteps, *actions[0].shape), dtype=np.float)
        np_rewards = np.empty(shape=(num_timesteps, 1), dtype=np.float)
        np_dones = np.empty(shape=(num_timesteps, 1), dtype=np.bool)

        # Process observation
        for timestep in range(num_timesteps):
            for image_i in range(args.num_images):
                np_observations[timestep, image_i * down_sampled_image_flattened_dim:(image_i + 1) * down_sampled_image_flattened_dim] = \
                    observation_transform(observations[timestep][image_i * original_image_flattened_dim:(image_i + 1) * original_image_flattened_dim])

            np_observations[timestep, -args.num_scalar_features:] = observations[timestep][-args.num_scalar_features:]
            np_actions[timestep] = actions[timestep]
            np_rewards[timestep] = rewards[timestep]
            np_dones[timestep] = dones[timestep]

        # This is for last observation
        for image_i in range(args.num_images):
            np_observations[num_timesteps, image_i * down_sampled_image_flattened_dim:(image_i + 1) * down_sampled_image_flattened_dim] = \
                observation_transform(observations[num_timesteps][image_i * original_image_flattened_dim:(image_i + 1) * original_image_flattened_dim])

        np_observations[num_timesteps, -args.num_scalar_features:] = observations[num_timesteps][-args.num_scalar_features:]

        cached_data_trajectories.append((np_observations, np_actions, np_rewards, np_dones))

    print("Saving cache to {}".format(args.cached_data_path))
    with open(args.cached_data_path, 'wb') as f:
        pickle.dump({"trajectories": cached_data_trajectories, "args": args}, f, protocol=4)


if __name__ == "__main__":
    args = parse_data_cache_args()
    construct_data_cache_from_sql(args)
