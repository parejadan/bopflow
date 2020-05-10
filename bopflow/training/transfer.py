import numpy as np

from bopflow.models.yolonet import BaseNet, yolo_v3
from bopflow import LOGGER


def reshape_mismatching_arrays(target_shapes, layer_weights):
    layer_depth = len(layer_weights)

    for depth in range(layer_depth):
        weights = layer_weights[depth]
        target_size = target_shapes[depth]
        if weights.shape != target_size:
            LOGGER.debug(
                f"Mismatch at depth {depth}. Expected: {target_size}\t| actual: {weights.shape}"
            )
            reshaped = np.resize(weights, target_size)
            layer_weights[depth] = reshaped


def force_fit_weights(source_weights, dest_layer):
    """
    For a given set of layer weights (source_weights) we iterate through it's depth
    and identify where the mismatch exists. The weights at mismatch depth are reshaped
    to the expected size in dest_layer at same depth. We then attempt to retransfer to
    dest_layer.
    """
    LOGGER.warning(
        f"Reshaping source weights for layer {dest_layer.name} to fit needed size"
    )
    target_shapes = [weights.shape for weights in dest_layer.get_weights()]
    reshape_mismatching_arrays(
        target_shapes=target_shapes, layer_weights=source_weights
    )
    LOGGER.debug(f"Attempting to set reshaped weights to: {dest_layer.name}")
    dest_layer.set_weights(source_weights)


def transfer_weights(source_layer, dest_layer, freeze=True):
    """
    Transfers weights from source_layer to destination layer. Force fits
    any mismatching weights from source to dest via weights reshape.
    """
    LOGGER.debug(f"Transfering weights for layer: {dest_layer.name}")
    source_weights = source_layer.get_weights()
    try:
        dest_layer.set_weights(source_weights)
    except ValueError:
        force_fit_weights(source_weights=source_weights, dest_layer=dest_layer)

    LOGGER.info("Transfer success")
    if freeze:
        LOGGER.info(f"Freezing layer: {dest_layer.name}")
        dest_layer.trainable = False


def transfer_layers(
    network: BaseNet,
    transfer_weights_path: str,
    transfer_weight_class_count: int,
    layer_transfer_cutoff=-1,
):
    """
    For given transfer weights considered to be pre-trained, we load them and transfer
    onto the network.model up to the specified layer_transfer_cutoff point.
    - Any weights that do not fit in the network.model.get_layers(<layer-name>)
      force fit them by reshaping to expected size with np.resize(weights, target_shape)
    - transfered layers are frozen to avoid weight adjustment during training

    Parameters
    ==========
    network: network consisting of the model we intend to transfer pre-trained
             weights onto.
    transfer_weights_path: *.tf record filepath consisting of pre-trained weights
                           we'll be working with
    transfer_weights_class_count: number of classes the pre-trained weights consist of.
                                  Idially this number should be less than the class count
                                  network.model was initialized on. If greater, the weights
                                  force fit reshape can still transfer them. But these force
                                  fitted weights could be considered inferior.
    layer_transfer_cutoff: cuttoff index in loaded weights layers we should stop the transfer process.
                           If only transfering n-1 layers, leave as default (-1)

    Returns
    =======
    Nothing. The network.model is modified directly.
    - NOTE: this does not compile the network.model, please make sure to do that before training
    """
    LOGGER.info(
        f"Loading weights onto network with class count: {transfer_weight_class_count}"
    )
    pretrained_model = yolo_v3(training=True, num_classes=transfer_weight_class_count)
    pretrained_model.load_weights(weights_path=transfer_weights_path)
    LOGGER.debug(f"Destination netowrk consist of layers: {network.layer_names}")
    for layer_name in network.layer_names[:layer_transfer_cutoff]:
        transfer_weights(
            source_layer=pretrained_model.get_layer(layer_name),
            dest_layer=network.model.get_layer(layer_name),
            freeze=True,
        )
