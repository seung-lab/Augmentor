from __future__ import print_function
import numpy as np

from .augment import Augment, Compose
import perturb


class Section(Augment):
    """Perturb sections in a training sample.

    Args:
        perturb (``Perturb``): ``Perturb`` class.
        max_sec (int): maximum number of sections to perturb.
        skip (float, optional): skip probability.
    """
    def __init__(self, perturb, max_sec, skip=0, **kwargs):
        self.perturb = perturb
        self.max_sec = max(max_sec, 0)
        self.skip = np.clip(skip, 0, 1)
        self.params = kwargs

    def __call__(self, sample, imgs=None, **kwargs):
        sample = Augment.to_tensor(sample)
        if np.random.rand() > self.skip:
            imgs, zdim = self._validate(sample, imgs)
            nsecs = np.random.randint(0, self.max_sec + 1)
            zlocs = np.random.choice(zdim, nsecs, replace=False)
            for z in zlocs:
                perturb = self.get_perturb()
                for key in imgs:
                    perturb(sample[key][...,z,:,:])
        return Augment.sort(sample)

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'max_sec={0}, '.format(self.max_sec)
        format_string += 'skip={:.3f}, '.format(self.skip)
        format_string += 'perturb={}, '.format(self.perturb)
        format_string += 'params={}'.format(self.params)
        format_string += ')'
        return format_string

    def get_perturb(self):
        return self.perturb(**self.params)

    def _validate(self, sample, keys):
        if keys is None:
            keys = sample.keys()
        zdims = {k: sample[k].shape[-3] for k in keys}
        zmax = max(zdims.values())
        zmin = min(zdims.values())
        assert zmax==zmin  # Do not allow inputs with different z-dim.
        assert zmax>self.max_sec
        return keys, zmax


class PartialSection(Section):
    """
    Perturb partial sections in a training sample.
    """
    def __init__(self):
        super(PartialSection, self).__init__(*args, **kwargs)

    def get_perturb(self):
        def perturb(img, rx, ry, quad, perturb):
            x = int(np.floor(rx * img.shape[-1]))
            y = int(np.floor(ry * img.shape[-2]))
            # 1st quadrant.
            if quad[0]:
                perturb[0](img[...,:y,:x])
            # 2nd quadrant.
            if quad[1]:
                perturb[1](img[...,y:,:x])
            # 3nd quadrant.
            if quad[2]:
                perturb[2](img[...,:y,x:])
            # 4nd quadrant.
            if quad[3]:
                perturb[3](img[...,y:,x:])
        rx, ry = np.random.rand(2)
        quad = np.random.rand(4) > 0.5
        perturb = [self.perturb(**self.params) for _ in range(4)]
        return lambda x: self._perturb(x)

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'prob={:.3f}, '.format(self.prob)
        format_string += 'skip={:.3f}, '.format(self.skip)
        format_string += 'perturb={}, '.format(self.perturb)
        format_string += 'params={}'.format(self.params)
        format_string += ')'
        return format_string
