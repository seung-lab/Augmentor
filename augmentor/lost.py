from __future__ import print_function
import numpy as np

from .augment import Augment, Compose


__all__ = ['LostSection']


class SingleLostSection(Augment):
    """Lost section augmentation.

    Args:
        skip (float, optional): skip probability.

    TODO:
        Support for valid architecture.
    """
    def __init__(self, skip=0,**kwargs):
        self.skip = np.clip(skip, 0, 1)
        self.zloc = 0

    def prepare(self, spec, **kwargs):
        # Biased coin toss
        if np.random.rand() < self.skip:
            self.zloc = 0
            return dict(spec)

        # Random sections
        zdim = self._validate(spec) - 1
        self.zloc = np.random.choice(zdim, 1, replace=False) + 1

        # Update spec
        for k, v in spec.items():
            pivot = -3
            zdim = v[pivot]
            spec[k] = v[:pivot] + (zdim+1,) + v[pivot+1:]
        return dict(spec)

    def __call__(self, sample, **kwargs):
        sample = Augment.to_tensor(sample)
        if self.zloc > 0:
            for k, v in sample.items():
                c, z, y, x = v.shape[-4:]
                w = np.zeros((c,z,y,x), dtype=v.dtype)
                w[:,:zloc,:,:] = v[:,:zloc,:,:]
                w[:,zloc:,:,:] = v[:,zloc+1:,:,:]
                sample[k] = w
        return Augment.sort(sample)

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'skip={:.2f}, '.format(self.skip)
        format_string += ')'
        return format_string

    def _validate(self, spec):
        zdims = [v[-3] for v in spec.values()]
        zmin, zmax = min(zdims), max(zdims)
        assert zmax == zmin
        assert zmax > 1
        return zmax


class LostSection(Compose):
    def __init__(self, maxsec, skip=0, **kwargs):
        augs = [SingleLostSection(skip=skip) for _ in range(maxsec)]
        super(LostSection, self).__init__(augs)


# class LostPlusMissing(Augment):
#     def __init__(self):
#         pass
