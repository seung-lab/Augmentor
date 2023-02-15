from __future__ import print_function
import numpy as np

import skimage.measure as measure

from .augment import Augment


class Label(Augment):
    """
    Recompute connected components.
    """
    def __init__(self, vec=False):
        self.vec = vec
        self.segs = []

    def prepare(self, spec, segs=[], **kwargs):
        self.segs = segs
        return dict(spec)

    def __call__(self, sample, **kwargs):
        sample = Augment.to_tensor(sample)        
        for k in self.segs:
            if k in sample:
                seg = sample[k][0,:,:,:].astype(np.uint32)
                split = measure.label(seg).astype(np.uint32)
                sample[k + '_split'] = split
                if self.vec:
                    sample[k + '_split_vec'] = self.vectorize(split)
        return Augment.sort(Augment.to_tensor(sample))

    def __repr__(self):
        format_string = self.__class__.__name__ + '()'
        return format_string

    def vectorize(self, seg):
        segs = []
        unq = np.unique(seg)
        for u in unq[1:]:
            segs.append((seg == u).astype(np.uint32))
        return np.stack(segs)
