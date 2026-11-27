"""Map project slices to the official SelfRDB training and sampling conventions."""
import torch
from torch.nn import functional as F

from t2flair.data import SliceDataset


def upstream_input(image):
    """Center-pad project intensities (already in [-1, 1]) to the official grid."""
    if image.shape[-2:] != (192, 192):
        raise ValueError(f"Expected project crop of 192x192, got {image.shape[-2:]}")
    return F.pad(image, (32, 32, 32, 32), value=-1.0)


def project_prediction(image):
    """Remove exactly the pad applied to each input, preserving slice order."""
    if image.shape[-2:] != (256, 256):
        raise ValueError(f"Expected SelfRDB output of 256x256, got {image.shape[-2:]}")
    return image[..., 32:224, 32:224].contiguous()


class BridgeSlices(SliceDataset):
    def __init__(self, processed, subjects):
        super().__init__(processed, subjects, flip=False)

    def __getitem__(self, index):
        volume_index, z = self.index[index]
        volume = self.vols[volume_index]
        def channel(key):
            return upstream_input(torch.from_numpy(volume[key][z]).float().unsqueeze(0))
        return channel("flair"), channel("t2"), index
