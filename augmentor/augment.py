from __future__ import print_function
from collections import OrderedDict
import numpy as np


class Augment(object):
    """
    Abstract interface.
    """
    def __init__(self):
        raise NotImplementedError

    def __call__(self, sample, **kwargs):
        raise NotImplementedError

    def prepare(self, spec, **kwargs):
        return dict(spec)

    def __repr__(self):
        raise NotImplementedError

    @staticmethod
    def sort(sample):
        # Ensure that sample is sorted by key.
        return OrderedDict(sorted(sample.items(), key=lambda x: x[0]))

    @staticmethod
    def to_tensor(data):
        """Ensure that data is a numpy 4D array."""
        assert isinstance(data, np.ndarray)
        if data.ndim == 2:
            data = data[np.newaxis,np.newaxis,...]
        elif data.ndim == 3:
            data = data[np.newaxis,...]
        elif data.ndim == 4:
            pass
        else:
            raise RuntimeError("data must be a numpy 4D array")
        assert data.ndim==4
        return data



class Compose(Augment):
    """Composes several augments together.

    Adapted from torchvision's transforms.Compose.

    Args:
        augments (list of ``Augment`` objects): list of augments to compose.

    Example:
        >>> augmentor.Compose([
        >>>     augmentor.Flip(axis=-1),
        >>>     augmentor.Flip(axis=-2),
        >>>     augmentor.Flip(axis=-3),
        >>>     augmentor.Transpose(axes=[0,1,3,2]),
        >>> ])
    """
    def __init__(self, augments):
        self.augments = augments

    def __call__(self, sample, **kwargs):
        for aug in self.augments:
            sample = aug(sample, **kwargs)
        return Augment.sort(sample)

    def prepare(self, spec, **kwargs):
        for aug in reversed(self.augments):
            spec = aug.prepare(spec, **kwargs)
        return dict(spec)

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for aug in self.augments:
            format_string += '\n'
            format_string += '    {0}'.format(aug)
        format_string += '\n)'
        return format_string
