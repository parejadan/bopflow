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
from bopflow.training.transfer import transfer_layers
from bopflow.iomanage import load_tfrecord_for_training
from bopflow import LOGGER


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

    transfer_layers(
        network=network,
        transfer_weights_path=args.weights,
        transfer_weight_class_count=80,
    )
    LOGGER.info("Compiling model")
    model.compile(optimizer=optimizer, loss=loss, run_eagerly=False)

    LOGGER.info(f"Initializing optimizer with learning rate {args.learning_rate}")
    optimizer = tf.keras.optimizers.Adam(lr=args.learning_rate)
    loss = [
        yolo_loss(anchors[mask], num_classes=args.new_model_class_count)
        for mask in anchor_masks
    ]

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
