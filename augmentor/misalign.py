from __future__ import print_function
import numpy as np

from .augment import Augment
from . import utils


class Misalign(Augment):
    """Translational misalignment.

    Args:
        max_disp (2-tuple of int): Min/max displacement.
        margin (int):
    """
    def __init__(self, disp=(5,25), margin=2):
        self.disp = disp
        self.margin = max(margin, 0)
        self.tx = 0
        self.ty = 0

    def prepare(self, spec, **kwargs):
        # Original spec
        self.spec = dict(spec)

        # Random displacement in x/y dimension.
        self.tx = np.random.randint(*self.disp)
        self.ty = np.random.randint(*self.disp)

        # Increase tensor dimension by the amount of displacement.
        zdims = dict()
        for k, shape in spec.items():
            z, y, x = shape[-3:]
            zdims[k] = z
            spec[k] = shape[:-2] + (y+self.ty, x+self.tx)

        # Pick a section to misalign.
        zmin = min(zdims.values())
        assert zmin >= 2*margin + 2
        zloc = np.random.randint(margin + 1, zmin - margin)

        # Offset z-location.
        self.zlocs = dict()
        for k, zdim in zdims.items():
            offset = zdim - zmin
            self.zlocs[k] = offset + zloc

        return dict(spec)

    def __call__(self, sample, **kwargs):
        return self._misalign(sample)

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'disp={0}, '.format(self.disp)
        format_string += 'margin={0}'.format(self.margin)
        format_string += ')'
        return format_string

    def _misalign(sample):
        sample = Augment.to_tensor(sample)

        for k, v in sample.items():
            # New tensor
            w = np.zeros(self.spec[k], dtype=v.dtype)
            w = utils.to_tensor(w)
            # Misalign.
            z, y, x = w.shape[-3:]
            zloc = self.zlocs[k]
            w[:,:zloc,...] = v[:,:zloc,:y,:x]
            w[:,zloc:,...] = v[:,zloc:,-y:,-x:]
            # Update sample.
            sample[k] = w

        return Augment.sort(sample)


class MisalignPlusMissing(Misalign):
    """
    Translational misalignment + one or two missing sections.
    """
    def __init__(self, **kwargs):
        super(MisalignPlusMissing, self).__init__(**kwargs)
        assert self.margin > 0

    def prepare(self, spec, **kwargs):
        spec = super(MisalignPlusMissing, self).prepare(spec, **kwargs)

        # Missing sections.
        self.nsec = 1 if np.random.rand() > 0.5 or 2

        # Interpolation.
        tx = round(self.tx / float(n+1))
        ty = round(self.ty / float(n+1))
        trans = [(tx + i*tx, ty + i*ty) for i in range(self.nsec)]

        return dict(spec)


class SlipMisalign(Augment):
    def __init__(self):
        raise NotImplementedError
