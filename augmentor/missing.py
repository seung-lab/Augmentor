from __future__ import print_function
import numpy as np

from .augment import Augment, Compose


class MissingSection(Augment):
    """Simulate missing sections in a training example.

    Each z-slice is independently chosen to be perturbed according to a
    user-specifed probability ``prob``.

    Args:
        prob (float):
        skip (float, optional):
        random_fill (bool):
    """
    def __init__(self, prob, skip=0, random_fill=True):
        self.prob = np.clip(prob, 0, 1)
        self.skip = np.clip(skip, 0, 1)
        self.random_fill = random_fill

    def __call__(self, sample, **kwargs):
        raise NotImplementedError

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'prob={:.3f}, '.format(self.prob)
        format_string += 'skip={:.3f}, '.format(self.skip)
        format_string += 'random_fill={}'.format(self.random_fill)
        format_string += ')'
        return format_string


class FullMissingSection(MissingSection):
    """
    Simulate full missing sections in a training example.
    """
    def __init__(self, **kwargs):
        super(FullMissingSection, self).__init__(**kwargs)

    def __call__(self, sample, imgs=None, **kwargs):
        if np.random.rand() > self.skip:
            if imgs is None:
                imgs = sample.keys()
            zdims = {k: sample[k].shape[-3] for k in imgs}
            zmax = max(zdims.values())
            zmin = min(zdims.values())
            assert zmax==zmin  # Do not allow inputs with different z-dim.
            for z in range(zmax):
                if np.random.rand() > self.prob:
                    continue
                # Random fill-out value.
                fill = np.random.rand() if self.random_fill else 0
                for key in imgs:
                    img = Augment.to_tensor(sample[key])
                    img[...,z,:,:] = fill
                    sample[key] = img
        return Augment.sort(sample)


class PartialMissingSection(MissingSection):
    """
    Simulate partial missing sections in a training example.
    """
    def __init__(self, **kwargs):
        super(PartialMissingSection, self).__init__(**kwargs)

    def __call__(self, sample, imgs=None, **kwargs):
        if np.random.rand() > self.skip:
            if imgs is None:
                imgs = sample.keys()
            zdims = {k: sample[k].shape[-3] for k in imgs}
            zmax = max(zdims.values())
            zmin = min(zdims.values())
            assert zmax==zmin  # Do not allow inputs with different z-dim.
            for z in range(zmax):
                if np.random.rand() > self.prob:
                    continue
                # Draw a random relative xy-coordinate.
                rx, ry = np.random.rand(2)
                # 1st quadrant.
                if np.random.rand() > 0.5:
                    # Random fill-out value.
                    fill = np.random.rand() if self.random_fill else 0
                    for key in imgs:
                        x = int(np.floor(rx * sample[key].shape[-1]))
                        y = int(np.floor(ry * sample[key].shape[-2]))
                        img = Augment.to_tensor(sample[key])
                        img[...,z,:y,:x] = fill
                        sample[key] = img
                # 2nd quadrant.
                if np.random.rand() > 0.5:
                    # Random fill-out value.
                    fill = np.random.rand() if self.random_fill else 0
                    for key in imgs:
                        x = int(np.floor(rx * sample[key].shape[-1]))
                        y = int(np.floor(ry * sample[key].shape[-2]))
                        img = Augment.to_tensor(sample[key])
                        img[...,z,y:,:x] = fill
                        sample[key] = img
                # 3nd quadrant.
                if np.random.rand() > 0.5:
                    # Random fill-out value.
                    fill = np.random.rand() if self.random_fill else 0
                    for key in imgs:
                        x = int(np.floor(rx * sample[key].shape[-1]))
                        y = int(np.floor(ry * sample[key].shape[-2]))
                        img = Augment.to_tensor(sample[key])
                        img[...,z,:y,x:] = fill
                        sample[key] = img
                # 4nd quadrant.
                if np.random.rand() > 0.5:
                    # Random fill-out value.
                    fill = np.random.rand() if self.random_fill else 0
                    for key in imgs:
                        x = int(np.floor(rx * sample[key].shape[-1]))
                        y = int(np.floor(ry * sample[key].shape[-2]))
                        img = Augment.to_tensor(sample[key])
                        img[...,z,y:,x:] = fill
                        sample[key] = img
        return Augment.sort(sample)


class MixedMissingSection(Compose):
    """
    Half 2D & half 3D.
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
