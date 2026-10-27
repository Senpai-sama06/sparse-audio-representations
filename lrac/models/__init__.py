from lrac.models.base import Representation
from lrac.models.dictionaries import FrameDictionary
from lrac.models.convolutional import ConvolutionalDictionary
from lrac.models.autoencoder import GroupedAutoencoder
from lrac.models.block_orthogonal import BlockOrthogonalDictionary
from lrac.models.sparse_autoencoder import SparseAutoencoder


def build_model(config, sample_rate):
    if config.kind == "sparse_autoencoder":
        return SparseAutoencoder(config, sample_rate)
    if config.kind == "block_orthogonal":
        return BlockOrthogonalDictionary(config, sample_rate)
    if config.kind == "convolutional":
        return ConvolutionalDictionary(config, sample_rate)
    if config.kind == "autoencoder":
        return GroupedAutoencoder(config, sample_rate)
    return FrameDictionary(config, sample_rate)
