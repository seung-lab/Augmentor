from __future__ import print_function
import numpy as np

import skimage.measure as measure

from .augment import Augment, Compose


class Label(Augment):
    """
    Recompute connected components.
    """
    def __init__(self, split_map=False):
        self.segs = []
        self.split_map = split_map

    def prepare(self, spec, segs=[], **kwargs):
        self.segs = self._validate(spec, segs)
        return dict(spec)

    def __call__(self, sample, **kwargs):
        sample = Augment.to_tensor(sample)
        for k in self.segs:
            seg = sample[k][0,:,:,:].astype(np.uint32)
            split = measure.label(seg).astype(np.uint32)
            sample[k] = split.astype(np.uint32)
            if self.split_map:
                groups = self.create_mapping(seg, split)
                sample[k + '_groups'] = groups
        return Augment.sort(Augment.to_tensor(sample))

    def create_mapping(self, seg, split):
        seg = seg.astype(np.uint64)
        split = split.astype(np.uint64)
        x = 2**32 * seg + split
        unq = np.unique(x)
        mapping = {}
        for u in unq:
            a, b = np.uint64(u // 2**32), np.uint64(u % 2**32)
            mapping[a] = mapping.get(a, []) + [b]
        groups = list(mapping.values())
        return groups

    def __repr__(self):
        format_string = self.__class__.__name__ + '()'
        return format_string

    def _validate(self, spec, segs):
        assert len(segs) > 0
        assert all(k in spec for k in segs)
        return segs
