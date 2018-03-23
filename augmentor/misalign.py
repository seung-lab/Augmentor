from __future__ import print_function
import numpy as np

from .augment import Augment, Blend
from . import utils


__all__ = ['Misalign','MisalignPlusMissing','SlipMisalign']


class Misalign(Augment):
    """Translational misalignment.

    Args:
        disp (2-tuple of int): Min/max displacement.
        margin (int):

    TODO:
        1. Valid architecture
        2. Augmentation territory
    """
    def __init__(self, disp, margin=0):
        self.disp = disp
        self.margin = max(margin, 0)
        self.tx = 0
        self.ty = 0
        self.zmin = 2

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
        assert zmin >= 2*self.margin + self.zmin
        zloc = np.random.randint(self.margin + 1, zmin - self.margin)

        # Offset z-location.
        self.zlocs = dict()
        for k, zdim in zdims.items():
            offset = zdim - zmin
            self.zlocs[k] = offset + zloc

        return dict(spec)

    def __call__(self, sample, **kwargs):
        return self.misalign(sample)

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'disp={0}, '.format(self.disp)
        format_string += 'margin={0}'.format(self.margin)
        format_string += ')'
        return format_string

    def misalign(self, sample):
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
            sample[k] = w

        return Augment.sort(sample)


class MisalignPlusMissing(Misalign):
    """
    Translational misalignment + missing section(s).
    """
    def __init__(self, disp, margin=1):
        margin = max(margin, 1)
        super(MisalignPlusMissing, self).__init__(disp, margin=margin)
        assert self.margin > 0
        self.imgs = []

    def prepare(self, spec, imgs=[], **kwargs):
        spec = super(MisalignPlusMissing, self).prepare(spec, **kwargs)
        self.both = np.random.rand() > 0.5
        self.imgs = self._validate(spec, imgs)
        return dict(spec)

    def __call__(self, sample, **kwargs):
        return self.misalign(sample, self.imgs)

    def _validate(self, spec, imgs):
        assert len(imgs) > 0
        assert all([k in spec for k in imgs])
        return imgs

    def misalign(self, sample, imgs):
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

            if k in imgs:
                # Missing section(s)
                w[:,zloc,...] = 0
                if self.both:
                    w[:,zloc-1,...] = 0
            else:
                # Target interpolation
                if self.both:
                    tx = round(self.tx / 3.0)
                    ty = round(self.ty / 3.0)
                    w[:,zloc-1,...] = v[:,zloc-1,ty:ty+y,tx:tx+x]
                    w[:,zloc,...] = v[:,zloc,-ty-y:-ty,-tx-x:-tx]
                else:
                    tx = round(self.tx / 2.0)
                    ty = round(self.ty / 2.0)
                    w[:,zloc,...] = v[:,zloc,ty:ty+y,tx:tx+x]

            # Update sample.
            sample[k] = w

        return Augment.sort(sample)


class SlipMisalign(Misalign):
    def __init__(self, disp, margin=1, interp=False):
        margin = max(margin, 1)
        super(SlipMisalign, self).__init__(disp, margin=margin)
        self.zmin = 1
        self.interp = interp
        self.imgs = []

    def prepare(self, spec, imgs=[], **kwargs):
        spec = super(SlipMisalign, self).prepare(spec, **kwargs)
        self.imgs = self._validate(spec, imgs)
        return dict(spec)

    def __call__(self, sample, **kwargs):
        return self.misalign(sample, self.imgs)

    def misalign(self, sample, imgs):
        sample = Augment.to_tensor(sample)

        for k, v in sample.items():
            # New tensor
            w = np.zeros(self.spec[k], dtype=v.dtype)
            w = utils.to_tensor(w)

            # Misalign.
            z, y, x = w.shape[-3:]
            zloc = self.zlocs[k]
            w[...] = v[...,:y,:x]
            if (k in imgs) or (not self.interp):
                w[:,zloc,...] = v[:,zloc,-y:,-x:]
            sample[k] = w

        return Augment.sort(sample)
