import os
import argparse
import tensorflow as tf
import numpy as np
import datetime
from tensorflow.keras.callbacks import (
    ReduceLROnPlateau,
    EarlyStopping,
    ModelCheckpoint,
    TensorBoard,
)

from bopflow.models.yolonet import yolo_v3, yolo_loss
from bopflow.iomanage import freeze_all, load_tfrecord_dataset
from bopflow.transform.image import transform_targets
from bopflow import LOGGER


def load_data(tfrecord_filepath, anchors, anchor_masks, batch_size):
    LOGGER.info(f"Loading dataset {tfrecord_filepath}")
    dataset = load_tfrecord_dataset(tfrecord_filepath)
    dataset = dataset.shuffle(buffer_size=512)
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(
        lambda img_raw, labels: (
            img_raw,
            transform_targets(labels, anchors, anchor_masks),
        )
    )

    return dataset


def reshape_mismatching_shapes(target_shapes, layer_weights):
    """
    For a list of target shapes (ordered) we iterate through each depth in layer_weights.
    At each depth we compare weights.shape against target_shape[depth_dex] check for mismatch.
    If mismatch exist we reshape the weights at that layer.
    """
    layer_depth = len(layer_weights)

    for dex in range(layer_depth):
        weights = layer_weights[dex]
        target = target_shapes[dex]
        if weights.shape != target:
            LOGGER.info(f"Mismatch found at layer depth {dex}")
            LOGGER.debug(f"Shape at depth {dex}: {weights.shape}\t| expected: {target}")
            reshaped = np.resize(weights, target)
            layer_weights[dex] = reshaped


def force_fit_weights(source_weights, dest_layer):
    """
    For a given set of layer weights (source_weights) we iterate through it's depth
    and identify where the mismatch exists. The weights at mismatch depth are reshaped
    to the expected size in dest_layer at same depth. We then attempt to retransfer to
    dest_layer.
    """
    LOGGER.warning(
        f"Reshaping layer [{dest_layer.name}] source weights to fit expected size"
    )
    target_shapes = [weights.shape for weights in dest_layer.get_weights()]
    reshape_mismatching_shapes(
        target_shapes=target_shapes, layer_weights=source_weights
    )
    LOGGER.debug(f"Attempting to set reshaped weights to layer [{dest_layer.name}]")
    dest_layer.set_weights(source_weights)


def transfer_weights(source_layer, dest_layer, freeze_layer):
    """
    Transfers weights from source_layer to destination layer. Force fits
    any mismatching weights from source to dest via weights reshape.
    """
    LOGGER.debug(f"Transfering weights for layer [{dest_layer.name}]")
    source_weights = source_layer.get_weights()
    try:
        dest_layer.set_weights(source_weights)
    except ValueError:
        force_fit_weights(source_weights=source_weights, dest_layer=dest_layer)
    LOGGER.info("Transfer success")
    if freeze_layer:
        LOGGER.info(f"Freezing layer [{dest_layer.name}]")
        freeze_all(dest_layer)
    else:
        LOGGER.info(f"Not freezing layer [{dest_layer.name}]")


def transfer_layers(network, transfer_weights_path, trained_class_count=80):
    """
    For the given network we load the pretrained weights onto each layer,
    except for the final layer as that's what will get adjusted.
    - the loaded layers get freezed to assure their integrity
    """
    LOGGER.info(f"Creating network with {trained_class_count} classes")
    model_pretrained = yolo_v3(training=True, num_classes=trained_class_count)
    model_pretrained.load_weights(transfer_weights_path)
    LOGGER.debug(f"Network consists of layers [{network.layer_names}]")
    for layer_name in network.layer_names[:-1]:
        transfer_weights(
            source_layer=model_pretrained.get_layer(layer_name),
            dest_layer=network.model.get_layer(layer_name),
            freeze_layer=True,
        )


def get_checkpoint_folder():
    run_date = datetime.datetime.now().strftime("%Y.%m.%d")
    output_path = f"checkpoints/{run_date}"
    os.makedirs(output_path, exist_ok=True)
    return output_path


def main(args):
    output_path = get_checkpoint_folder()
    physical_devices = tf.config.experimental.list_physical_devices("GPU")
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    LOGGER.info(f"Creating model to train with {args.new_model_class_count} classes")
    network = yolo_v3(
        training=True, num_classes=args.new_model_class_count, just_model=False
    )
    anchors, anchor_masks, model = network.anchors, network.masks, network.model

    train_dataset = load_data(
        tfrecord_filepath=args.tfrecord_train,
        anchors=anchors,
        anchor_masks=anchor_masks,
        batch_size=args.batch_size,
    )
    train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    val_dataset = load_data(
        tfrecord_filepath=args.tfrecord_test,
        anchors=anchors,
        anchor_masks=anchor_masks,
        batch_size=args.batch_size,
    )

    transfer_layers(network=network, transfer_weights_path=args.weights)
    LOGGER.info(f"Initializing optimizer with learning rate {args.learning_rate}")
    optimizer = tf.keras.optimizers.Adam(lr=args.learning_rate)
    loss = [
        yolo_loss(anchors[mask], num_classes=args.new_model_class_count)
        for mask in anchor_masks
    ]

    LOGGER.info("Compiling model")
    model.compile(optimizer=optimizer, loss=loss, run_eagerly=False)

    LOGGER.info(f"Defining checkpoints for output {output_path}")
    callbacks = [
        ReduceLROnPlateau(verbose=1),
        EarlyStopping(patience=3, verbose=1),
        ModelCheckpoint(
            output_path + "/yolov3_train_{epoch}.tf", verbose=1, save_weights_only=True
        ),
        TensorBoard(log_dir="logs"),
    ]

    LOGGER.info("Initiating model fitting process")
    model.fit(
        train_dataset,
        epochs=args.epochs,
        callbacks=callbacks,
        validation_data=val_dataset,
    )

    trained_weights_path = f"{output_path}/weights.tf"
    LOGGER.info(
        f"Training complete. Saving trained model weights to {trained_weights_path}"
    )
    model.save_weights(trained_weights_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="For fine tuning yolov3 object detection against new object classes"
    )

    parser.add_argument("-tfrecord-train", default="", help="path to training dataset")
    parser.add_argument("-tfrecord-test", default="", help="path to testing dataset")
    parser.add_argument(
        "--weights", default="./checkpoints/yolov3.tf", help="path to weights file"
    )
    parser.add_argument("--epochs", default=2, help="number of epochs")
    parser.add_argument("--batch-size", default=8, help="batch size")
    parser.add_argument("--learning-rate", default=1e-4, help="learning rate")
    parser.add_argument(
        "--new-model-class-count",
        default=81,
        help="class count resulting model should consist of. If adding 1 more, pass 81",
    )
    args = parser.parse_args()

    args.epochs = int(args.epochs)
    args.batch_size = int(args.batch_size)
    args.learning_rate = float(args.learning_rate)
    args.new_model_class_count = int(args.new_model_class_count)
    main(args)
