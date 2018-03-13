from __future__ import print_function
import collections

from .augment import Augment, Compose
from .section import Section, PartialSection
from . import perturb


class MissingSection(Section):
    """
    Simulate full missing sections in a training sample.
    """
    def __init__(self, max_sec, skip=0, value=0, random=True):
        super(MissingSection, self).__init__(perturb.Fill, max_sec, skip=skip)
        self.params = dict(value=value, random=random)


class PartialMissingSection(PartialSection):
    """
    Simulate partial missing sections in a training sample.
    """
    def __init__(self, max_sec, skip=0, value=0, random=True):
        super(PartialMissingSection, self).__init__(
            perturb.Fill, max_sec, skip=skip
        )
        self.params = dict(value=value, random=random)


class MixedMissingSection(Compose):
    """
    Mixed full & partial missing sections.
    """
    def __init__(self, max_sec, **kwargs):
        if isinstance(max_sec, collections.Sequence):
            assert len(max_sec) == 2
            full = MissingSection(max_sec[0], **kwargs)
            part = PartialMissingSection(max_sec[1], **kwargs)
        else:
            full = MissingSection(max_sec, **kwargs)
            part = PartialMissingSection(max_sec, **kwargs)
        super(MixedMissingSection, self).__init__([full,part])


########################################################################
## Testing.
########################################################################
if __name__ == "__main__":
    import numpy as np
    
    full  = MissingSection(2)
    part  = PartialMissingSection(2)
    mixed = MixedMissingSection((1,1))

    print(full)
    print(part)
    print(mixed)

    sample = dict(input=np.random.rand(3,3,3))
    print('\nsample = {}'.format(sample))
    print(mixed(sample))
