from __future__ import print_function
import collections

from .augment import Augment, Compose
from .section import Section, PartialSection
from . import perturb


class BlurrySection(Section):
    """
    Simulate full out-of-focus sections in a training sample.
    """
    def __init__(self, max_sec, skip=0, sigma=5.0, random=True):
        super(BlurrySection, self).__init__(perturb.Blur, max_sec, skip=skip)
        self.params = dict(sigma=sigma, random=random)


class PartialBlurrySection(PartialSection):
    """
    Simulate partial out-of-focus sections in a training sample.
    """
    def __init__(self, max_sec, skip=0, sigma=5.0, random=True):
        super(PartialBlurrySection, self).__init__(
            perturb.Blur, max_sec, skip=skip
        )
        self.params = dict(sigma=sigma, random=random)


class MixedBlurrySection(Compose):
    """
    Mixed full & partial out-of-focus sections.
    """
    def __init__(self, max_sec, **kwargs):
        if isinstance(max_sec, collections.Sequence):
            assert len(max_sec) == 2
            full = BlurrySection(max_sec[0], **kwargs)
            part = PartialBlurrySection(max_sec[1], **kwargs)
        else:
            full = BlurrySection(max_sec, **kwargs)
            part = PartialBlurrySection(max_sec, **kwargs)
        super(MixedBlurrySection, self).__init__([full,part])


########################################################################
## Testing.
########################################################################
if __name__ == "__main__":
    import numpy as np

    full  = BlurrySection(2)
    part  = PartialBlurrySection(2)
    mixed = MixedBlurrySection((1,1))

    print(full)
    print(part)
    print(mixed)

    sample = dict(input=np.random.rand(3,3,3))
    print('\nsample = {}'.format(sample))
    print(full(sample))

    sample = dict(input=np.random.rand(3,3,3))
    print('\nsample = {}'.format(sample))
    print(part(sample))

    sample = dict(input=np.random.rand(3,3,3))
    print('\nsample = {}'.format(sample))
    print(mixed(sample))
