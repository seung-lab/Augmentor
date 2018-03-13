from __future__ import print_function
import numpy as np

from .augment import Augment
from .warping import warping


class Warp(Augment):
    """
    Warping by combining 5 types of linear transformations:
        1. Continuous rotation
        2. Shear
        3. Twist
        4. Scale
        5. Perspective stretch
    """
    def __init__(self, skip=0):
        self.skip = np.clip(skip, 0, 1)

    def prepare(self, spec, **kwargs):
        raise NotImplementedError

    def __call__(self, sample, keys=None, **kwargs):
        sample = Augment.to_tensor(sample)
        # Biased coin toss.
        if np.random.rand() > self.skip:
            pass
        return Augment.sort(sample)

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'skip={:.2f}'.format(self.skip)
        format_string += ')'
        return format_string
