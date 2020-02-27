from cv2 import cv2
import numpy as np
import _pickle as pkl
import torch
from torch.distributions import normal

class Normalize:
    def __init__(self, mean, var):
        self.mean = mean
        self.var = var

    def __call__(self, x):
        return (x - self.mean) / self.var

    def __repr__(self):
        return self.__class__.__name__ + '(mean={self.mean}, var={self.var})'

class Dropped(object):
    """Drop an image measurement (set image as 1).

    Args:
        p: Probability of applying this transform
    """
    def __init__(self, p=1):
        self.p = p

    def __call__(self, img):
        if np.random.binomial(1, self.p):
            black_out_img = torch.zeros_like(img)
            black_out_img = torch.clamp(black_out_img, 0, 1)
            return black_out_img
        else:
            return img

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)

class GaussianNoise(object):
    """Add Gaussian noise to the image.

    Args:
        p: Probability of applying this transform
        std: Standard deviation of Gaussian noise
        mean: Mean of Gaussian noise
    """
    def __init__(self, p=1, std=0.1, mean=0.0):
        self.p = p
        self.n = normal.Normal(mean, std)

    def __call__(self, img):
        if np.random.binomial(1,self.p):
            noise = torch.abs(self.n.sample((img.shape[0], img.shape[1]))).to(device=img.device)
            noisy_img = img + noise
            noisy_img_clipped = torch.clamp(noisy_img, 0, 1)
            return noisy_img_clipped
        else:
            return img

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)

class Obstruct(object):
    """Obstruct an image.

    Args:
        p: Probability of applying this transform
    """
    def __init__(self, p=0.5, value=0):
        self.p = p
        self.value = value

    def __call__(self, img):
        if np.random.binomial(1,self.p):
            obstructed_img = img.clone()
            size = np.random.randint(12,48, size=2)
            location = np.random.randint(0,24, size=2)
            x = location[0]
            y = location[1]
            obstructed_img[x:x+size[0], y:y+size[1]] = self.value
            return obstructed_img
        else:
            return img
            
    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)

class DownSample():
    def __init__(self, height, width):
        self._height = height
        self._width = width

    def __call__(self, image):
        image = cv2.resize(image, (self._height, self._width))
        return image


class NormalizeImage():
    def __init__(self, const=255):
        self.const = const

    def __call__(self, image):
        return image / self.const


class AsType():
    def __init__(self, dtype=np.uint8):
        self.dtype = dtype

    def __call__(self, input_data):
        return input_data.astype(self.dtype)


class Reshape():
    def __init__(self, shape):
        self.shape = shape

    def __call__(self, input_data):
        return input_data.reshape(self.shape)


class Transpose():
    def __init__(self, transpose_shape):
        self.transpose_shape = transpose_shape

    def __call__(self, input_data):
        return input_data.transpose(self.transpose_shape)


class DropScalarFeature():
    def __init__(self, scalar_feature_dim):
        self.scalar_feature_dim = scalar_feature_dim

    def __call__(self, input_data):
        return input_data[:, :-self.scalar_feature_dim]
