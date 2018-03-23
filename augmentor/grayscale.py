from __future__ import print_function
import numpy as np

from .augment import Augment, Blend
from .perturb import Grayscale
from .section import Section


class Grayscale3D(Augment):
    """Grayscale value perturbation.

    Randomly adjust contrast/brightness, and apply random gamma correction.
    """
    def __init__(self, contrast_factor=0.3, brightness_factor=0.3, skip=0.3):
        self.contrast_factor = contrast_factor
        self.brightness_factor = brightness_factor
        self.skip = np.clip(skip, 0, 1)
        self.do_aug = False
        self.imgs = []

    def prepare(self, spec, imgs=[], **kwargs):
        # Biased coin toss.
        self.do_aug = np.random.rand() > self.skip
        self.imgs = self._validate(spec, imgs)
        return dict(spec)

    def __call__(self, sample, **kwargs):
        sample = Augment.to_tensor(sample)
        if self.do_aug:
            perturb = Grayscale(self.contrast_factor,
                                self.brightness_factor)
            for k in self.imgs:
                perturb(sample[k])
        return Augment.sort(sample)

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'contrast_factor={0}, '.format(self.contrast_factor)
        format_string += 'brightness_factor={0}, '.format(self.brightness_factor)
        format_string += 'skip={:.2f}'.format(self.skip)
        format_string += ')'
        return format_string

    def _validate(self, spec, imgs):
        assert len(imgs) > 0
        assert all([k in spec for k in imgs])
        return imgs


class Grayscale2D(Section):
    """
    Perturb each z-slice independently.
    """
    def __init__(self, contrast_factor=0.3, brightness_factor=0.3, skip=0.3):
        super(Grayscale2D, self).__init__(Grayscale, 0, prob=1, skip=skip)
        self.params = dict(contrast_factor=contrast_factor,
                           brightness_factor=brightness_factor)


class GrayscaleMixed(Blend):
    """
    Half 2D & half 3D.
    """
    def __init__(self, **kwargs):
        grayscales = [Grayscale2D(**kwargs), Grayscale3D(**kwargs)]
        super(GrayscaleMixed, self).__init__(grayscales)


# class Grayscale2D(Section):
#     """
#     Perturb each z-slice independently.
#     """
#     def __init__(self, contrast_factor=0.3, brightness_factor=0.3, skip=0.3):
#         super(Grayscale2D, self).__init__(Grayscale, 0, prob=1, skip=skip)
#         self.params = dict(contrast_factor=contrast_factor,
#                            brightness_factor=brightness_factor)


########################################################################
## Testing.
########################################################################
if __name__ == "__main__":

    gray2D = Grayscale2D(skip=0)
    gray3D = Grayscale3D(skip=0)
    graymixed = GrayscaleMixed(skip=0)

    print(gray2D)
    print(gray3D)
    print(graymixed)

    sample = dict(input=np.random.rand(2,2,2))
    print('sample = {}'.format(sample))

    print(gray2D(sample))
    print(gray3D(sample))
    print(graymixed(sample))
    print(graymixed(sample))
    print(graymixed(sample))
