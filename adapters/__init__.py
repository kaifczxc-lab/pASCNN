"""Task adapters for pASCNN."""

from pascnn.adapters.mnist import MNISTAdapter
from pascnn.adapters.mnist import MNIST_CLASS_CODEBOOK
from pascnn.adapters.mnist import MNISTEncoder
from pascnn.adapters.mnist import MNIST_IMAGE_HEIGHT
from pascnn.adapters.mnist import MNIST_IMAGE_WIDTH
from pascnn.adapters.mnist import MNIST_NUM_CLASSES
from pascnn.adapters.mnist import MNIST_NUM_LOGIC_HEADS
from pascnn.adapters.mnist import MNISTDecodeOutputs
from pascnn.adapters.sequence import MaskedMeanSequenceAdapter
from pascnn.adapters.sequence import MaskedTokenwiseSequenceAdapter
from pascnn.adapters.sequence import SequenceBatch
from pascnn.adapters.sequence import TokenwiseSequenceEmbedding
from pascnn.adapters.sequence import reshape_logical_readout_to_sequence

__all__ = [
    "MaskedMeanSequenceAdapter",
    "MaskedTokenwiseSequenceAdapter",
    "MNISTAdapter",
    "MNISTDecodeOutputs",
    "MNIST_CLASS_CODEBOOK",
    "MNISTEncoder",
    "MNIST_IMAGE_HEIGHT",
    "MNIST_IMAGE_WIDTH",
    "MNIST_NUM_CLASSES",
    "MNIST_NUM_LOGIC_HEADS",
    "SequenceBatch",
    "TokenwiseSequenceEmbedding",
    "reshape_logical_readout_to_sequence",
]
