from __future__ import print_function
import numpy as np

from .augment import Augment, Compose
from .section import FullSection, PartialSection
import perturb


class FullMissingSection(FullSection):
    """
    Simulate full missing sections in a training sample.
    """
    def __init__(self, prob, skip=0, value=0, random=True):
        super(FullMissingSection, self).__init__(prob, skip)
        self.perturb = perturb.Fill
        self.params = dict(value=value, random=random)


class PartialMissingSection(PartialSection):
    """
    Simulate partial missing sections in a training sample.
    """
    def __init__(self, prob, skip=0, value=0, random=True):
        super(PartialMissingSection, self).__init__(prob, skip)
        self.perturb = perturb.Fill
        self.params = dict(value=value, random=random)


class MixedMissingSection(Compose):
    """
    Mixed full & partial missing sections.
    """
    def __init__(self, **kwargs):
        full = FullMissingSection(**kwargs)
        part = PartialMissingSection(**kwargs)
        super(MixedMissingSection, self).__init__([full,part])


########################################################################
## Testing.
########################################################################
if __name__ == "__main__":

    full  = FullMissingSection(prob=0.5)
    part  = PartialMissingSection(prob=0.5)
    mixed = MixedMissingSection(prob=0.5)

    print(full)
    print(part)
    print(mixed)

    for _ in range(1):
        sample = dict(input=np.random.rand(3,3,3))
        print('\nsample = {}'.format(sample))
        print(mixed(sample))
