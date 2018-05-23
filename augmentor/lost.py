from __future__ import print_function
import numpy as np

from .augment import Augment, Compose


__all__ = ['LostSection', 'LostPlusMissing']


class LostSection(Augment):
    """Lost section augmentation.

    Args:
        nsec: number of consecutive lost sections.
        skip (float, optional): skip probability.

    TODO:
        Support for valid architecture.
    """
    def __init__(self, nsec, skip=0,**kwargs):
        self.nsec = max(nsec, 1)
        self.skip = np.clip(skip, 0, 1)
        self.zloc = 0

    def prepare(self, spec, **kwargs):
        # Biased coin toss
        if np.random.rand() < self.skip:
            self.zloc = 0
            return dict(spec)

        # Random sections
        zdim = self._validate(spec) - 1
        zloc = np.random.choice(zdim, 1, replace=False) + 1
        self.zloc = zloc[0]

        # Update spec
        spec = dict(spec)
        for k, v in spec.items():
            pivot = -3
            new_z = v[pivot] + self.nsec
            spec[k] = v[:pivot] + (new_z,) + v[pivot+1:]
        return spec

    def __call__(self, sample, **kwargs):
        sample = Augment.to_tensor(sample)
        if self.zloc > 0:
            nsec = self.nsec
            zloc = self.zloc
            for k, v in sample.items():
                c, z, y, x = v.shape[-4:]
                w = np.zeros((c, z - nsec, y, x), dtype=v.dtype)
                w[:,:zloc,:,:] = v[:,:zloc,:,:]
                w[:,zloc:,:,:] = v[:,zloc+nsec:,:,:]
                sample[k] = w
        return Augment.sort(sample)

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'nsec={:}, '.format(self.nsec)
        format_string += 'skip={:.2f}, '.format(self.skip)
        format_string += ')'
        return format_string

    def _validate(self, spec):
        zdims = [v[-3] for v in spec.values()]
        zmin, zmax = min(zdims), max(zdims)
        assert zmax == zmin
        assert zmax > 1
        return zmax


class LostPlusMissing(LostSection):
    def __init__(self, skip=0, value=0, random=False):
        super(LostPlusMissing, self).__init__(3, skip=skip)
        self.value = value
        self.random = random
        self.imgs = []

    def prepare(self, spec, imgs=[], **kwargs):
        spec = super(LostPlusMissing, self).prepare(spec)
        assert len(imgs) > 0
        assert all(k in spec for k in imgs)
        self.imgs = imgs
        return dict(spec)

    def __call__(self, sample, **kwargs):
        return self.augment(sample)

    def augment(self, sample):
        sample = Augment.to_tensor(sample)

        if self.zloc > 0:
            val = np.random.rand() if self.random else self.value
            assert self.nsec == 3
            nsec = self.nsec
            zloc = self.zloc
            for k, v in sample.items():
                # New tensor
                c, z, y, x = v.shape[-4:]
                w = np.zeros((c, z - nsec + 1, y, x), dtype=v.dtype)

                # Non-missing part
                w[:,:zloc,:,:] = v[:,:zloc,:,:]
                w[:,zloc+1:,:,:] = v[:,zloc + nsec:,:,:]

                # Missing part
                if k in self.imgs:
                    w[:,zloc,...] = val
                else:
                    w[:,zloc,...] = v[:,zloc+1,:,:]

                # Update sample
                sample[k] = w

        return Augment.sort(sample)

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'nsec={:}, '.format(self.nsec)
        format_string += 'skip={:.2f}, '.format(self.skip)
        format_string += 'value={:}, '.format(self.value)
        format_string += 'random={:}, '.format(self.random)
        format_string += ')'
        return format_string
