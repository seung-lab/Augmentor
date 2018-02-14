from __future__ import print_function
import numpy as np
from scipy.ndimage.filters import gaussian_filter

from .augment import Augment, Compose


class BlurrySection(Augment):
    """Simulate out-of-focus (blurry) sections in a training sample.

    Each z-slice is independently chosen to be perturbed according to a
    user-specifed probability ``prob``.

    Args:
        prob (float):
        skip (float, optional):
        sigma_max (float, optional):
    """
    def __init__(self, prob, skip=0, sigma_max=5.0):
        self.prob = np.clip(prob, 0, 1)
        self.skip = np.clip(skip, 0, 1)
        self.sigma_max = max(sigma_max, 0.0)

    def __call__(self, sample, **kwargs):
        raise NotImplementedError

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'prob={:.3f}, '.format(self.prob)
        format_string += 'skip={:.3f}, '.format(self.skip)
        format_string += 'sigma_max={:.1f}'.format(self.sigma_max)
        format_string += ')'
        return format_string


class FullBlurrySection(BlurrySection):
    """
    Simulate full out-of-focus (blurry) sections in a training example.
    """
    def __init__(self, **kwargs):
        super(FullBlurrySection, self).__init__(**kwargs)

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
                sigma = np.random.rand() * self.sigma_max
                for key in imgs:
                    img = Augment.to_tensor(sample[key])
                    img[...,z,:,:] = gaussian_filter(img, sigma=sigma)
                    sample[key] = img
        return Augment.sort(sample)


class PartialBlurrySection(BlurrySection):
    """
    Simulate partial out-of-focus (blurry) sections in a training example.
    """
    def __init__(self, **kwargs):
        super(PartialBlurrySection, self).__init__(**kwargs)

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


class MixedBlurrySection(Compose):
    """
    Mixed full & partial out-of-focus (blurry) sections.
    """
    def __init__(self, **kwargs):
        full = FullBlurrySection(**kwargs)
        part = PartialBlurrySection(**kwargs)
        super(MixedBlurrySection, self).__init__([full,part])


########################################################################
## Testing.
########################################################################
if __name__ == "__main__":

    full  = FullBlurrySection(prob=0.5)
    part  = PartialBlurrySection(prob=0.5)
    mixed = MixedBlurrySection(prob=0.5)

    print(full)
    print(part)
    print(mixed)

    for _ in range(1):
        sample = dict(input=np.random.rand(3,3,3))
        print('\nsample = {}'.format(sample))
        print(mixed(sample))
