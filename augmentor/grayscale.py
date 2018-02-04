from __future__ import print_function
import numpy as np

from .augment import Augment, Blend


def perturb_image(img, contrast_factor, brightness_factor):
    """In-place random grayscale value perturbation."""
    img *= 1 + (np.random.rand() - 0.5) * contrast_factor
    img += (np.random.rand() - 0.5) * brightness_factor
    np.clip(img, 0, 1, out=img)
    img **= 2.0**(np.random.rand()*2 - 1)
    return img


class Grayscale(Augment):
    """Grayscale value perturbation.

    Randomly adjust contrast/brightness, and apply random gamma correction.
    """
    def __init__(self, contrast_factor=0.3, brightness_factor=0.3, skip=0.3):
        self.contrast_factor = contrast_factor
        self.brightness_factor = brightness_factor
        self.skip = np.clip(skip, 0, 1)

    def __call__(self, sample, **kwargs):
        raise NotImplementedError

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'contrast_factor={0}, '.format(self.contrast_factor)
        format_string += 'brightness_factor={0}, '.format(self.brightness_factor)
        format_string += 'skip={:.3f}'.format(self.skip)
        format_string += ')'
        return format_string


class Grayscale2D(Grayscale):
    """
    Perturb each z-slice independently.
    """
    def __init__(self, **kwargs):
        super(Grayscale2D, self).__init__(**kwargs)

    def __call__(self, sample, imgs=None, **kwargs):
        # Biased coin toss.
        if np.random.rand() > self.skip:
            if imgs is None:
                imgs = sample.keys()
            for key in imgs:
                img = Augment.to_tensor(sample[key])
                for z in range(img.shape[-3]):
                    z_slice = img[...,z,:,:]
                    perturb_image(z_slice, self.contrast_factor,
                                           self.brightness_factor)
        return Augment.sort(sample)


class Grayscale3D(Grayscale):
    """
    Perturb every z-slice identically.
    """
    def __init__(self, **kwargs):
        super(Grayscale3D, self).__init__(**kwargs)

    def __call__(self, sample, imgs=None, **kwargs):
        # Biased coin toss.
        if np.random.rand() > self.skip:
            if imgs is None:
                imgs = sample.keys()
            for key in imgs:
                img = Augment.to_tensor(sample[key])
                perturb_image(img, self.contrast_factor,
                                   self.brightness_factor)
        return Augment.sort(sample)


class GrayscaleMixed(Blend):
    """
    Half 2D & half 3D.
    """
    def __init__(self, **kwargs):
        grays = [Grayscale2D(**kwargs), Grayscale3D(**kwargs)]
        super(GrayscaleMixed, self).__init__(grays)


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
