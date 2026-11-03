from approaches.representations.models.base import Representation
from approaches.representations.models.dictionaries import FrameDictionary
from approaches.representations.models.convolutional import ConvolutionalDictionary
from approaches.representations.models.autoencoder import GroupedAutoencoder
from approaches.representations.models.block_orthogonal import BlockOrthogonalDictionary
from approaches.representations.models.sparse_autoencoder import SparseAutoencoder


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
