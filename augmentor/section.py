from __future__ import print_function
import numpy as np

from .augment import Augment, Compose
from .perturb import Perturb


class Section(Augment):
    """Perturb random sections in a training sample.

    Args:
        perturb (``Perturb``): ``Perturb`` class.
        max_sec (int): maximum number of sections to perturb.
        skip (float, optional): skip probability.
    """
    def __init__(self, perturb, max_sec, skip=0, **params):
        assert issubclass(perturb, Perturb)
        self.perturb = perturb
        self.max_sec = max(max_sec, 0)
        self.skip = np.clip(skip, 0, 1)
        self.params = params

    def __call__(self, sample, keys=None, **kwargs):
        sample = Augment.to_tensor(sample)
        if np.random.rand() > self.skip:
            keys, zdim = self._validate(sample, keys)
            nsecs = np.random.randint(1, self.max_sec + 1)
            zlocs = np.random.choice(zdim, nsecs, replace=False)
            for z in zlocs:
                perturb = self.get_perturb()
                for k in keys:
                    perturb(sample[k][...,z,:,:])
        return Augment.sort(sample)

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'max_sec={}, '.format(self.max_sec)
        format_string += 'skip={:.2f}, '.format(self.skip)
        format_string += 'perturb={}, '.format(self.perturb)
        format_string += 'params={}'.format(self.params)
        format_string += ')'
        return format_string

    def get_perturb(self):
        return self.perturb(**self.params)

    def _validate(self, sample, keys):
        if keys is None:
            keys = sample.keys()
        zdims = [sample[k].shape[-3] for k in keys]
        zmin, zmax = min(zdims), max(zdims)
        assert zmax==zmin  # Do not allow inputs with different z-dim.
        assert zmax>self.max_sec
        return keys, zmax


class PartialSection(Section):
    """
    Perturb partially random sections in a training sample.

    TODO:
        1. margin
    """
    def get_perturb(self):
        class _PerturbQuadrant():
            def __init__(self, perturb, rx, ry, quad):
                self.perturb = perturb
                self.rx = rx
                self.ry = ry
                self.quad = quad

            def __call__(self, img):
                x = int(np.floor(self.rx * img.shape[-1]))
                y = int(np.floor(self.ry * img.shape[-2]))
                # 1st quadrant.
                if self.quad[0]:
                    self.perturb[0](img[...,:y,:x])
                # 2nd quadrant.
                if self.quad[1]:
                    self.perturb[1](img[...,y:,:x])
                # 3nd quadrant.
                if self.quad[2]:
                    self.perturb[2](img[...,:y,x:])
                # 4nd quadrant.
                if self.quad[3]:
                    self.perturb[3](img[...,y:,x:])

        rx, ry = np.random.rand(2)
        quad = np.random.rand(4) > 0.5
        perturb = [self.perturb(**self.params) for _ in range(4)]
        return _PerturbQuadrant(perturb, rx, ry, quad)


class MixedSection(PartialSection):
    def get_perturb(self):
        if np.random.rand() > 0.5:
            return Section.get_perturb(self)
        else:
            return PartialSection.get_perturb(self)
