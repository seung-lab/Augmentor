from __future__ import print_function
import numpy as np
from scipy.ndimage.filters import gaussian_filter


class Perturb(object):
    """
    Callable class for in-place image perturbation.
    """
    def __init__(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, img):
        raise NotImplementedError

    def __repr__(self):
        return self.__class__.__name__


class Grayscale(Perturb):
    """Grayscale value perturbation."""
    def __init__(self, contrast_factor=0.3, brightness_factor=0.3):
        contrast_factor = np.clip(contrast_factor, 0, 2)
        brightness_factor = np.clip(brightness_factor, 0, 2)
        param = dict()
        param['contrast'] = 1 + (np.random.rand() - 0.5) * contrast_factor
        param['brightness'] = (np.random.rand() - 0.5) * brightness_factor
        param['gamma'] = (np.random.rand()*2 - 1)
        self.param = param

    def __call__(self, img):
        img *= self.param['contrast']
        img += self.param['brightness']
        np.clip(img, 0, 1, out=img)
        img **= 2.0**self.param['gamma']


class Fill(Perturb):
    """Fill with a scalar value."""
    def __init__(self, value=0, random=False):
        value = np.clip(value, 0, 1)
        self.value = np.random.rand() if random else value

    def __call__(self, img):
        img = self.value


class Blur(Perturb):
    """Gaussian blurring."""
    def __init__(self, sigma, random=False):
        sigma = max(sigma, 0)
        self.sigma = np.random.rand()*sigma if random else sigma

    def __call__(self, img):
        gaussian_filter(img, sigma=sigma, output=img)