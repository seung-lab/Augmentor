import imgaug.augmenters as iaa
import numpy as np

from .augment import Augment


__all__ = ['AdditiveGaussianNoise']


class AdditiveGaussianNoise(Augment):
    """Additive Gaussian noise.
    """
    def __init__(self, sigma=(0.01,0.1), per_channel=False):
        self.sigma = sigma
        self.per_channel = per_channel
        self.aug = iaa.AdditiveGaussianNoise(scale=sigma, 
                                             per_channel=per_channel)
        self.imgs = []

    def prepare(self, spec, imgs=[], **kwargs):
        self.imgs = self._validate(spec, imgs)
        return dict(spec)

    def __call__(self, sample, **kwargs):
        sample = Augment.to_tensor(sample)
        for k in self.imgs:
            sample[k] = np.clip(self.aug(images=sample[k]), 0, 1)
        return Augment.sort(sample)

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += f'sigma={self.sigma}, '
        format_string += f'per_channel={self.per_channel}'
        format_string += ')'
        return format_string

    def _validate(self, spec, imgs):
        assert len(imgs) > 0
        assert all(k in spec for k in imgs)
        return imgs