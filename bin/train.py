import os
import argparse
import tensorflow as tf
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


def load_model_with_ancors(num_classes, use_tiny):
    network = yolo_v3(
        training=True, num_classes=num_classes, use_tiny=use_tiny, just_model=False
    )

    return network.anchors, network.masks, network.model


def load_data(tfrecord_filepath, anchors, anchor_masks, batch_size):
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


def transfer_darknet_layer(model, transfer_weights_path, use_tiny):
    model_pretrained = yolo_v3(training=True, num_classes=80, use_tiny=use_tiny)
    model_pretrained.load_weights(transfer_weights_path)
    model.get_layer("yolo_darknet").set_weights(
        model_pretrained.get_layer("yolo_darknet").get_weights()
    )
    freeze_all(model.get_layer("yolo_darknet"))


def get_checkpoint_folder():
    run_date = datetime.datetime.now().strftime("%Y.%m.%d")
    output_dir = f"checkpoints/{run_date}"
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def main(args):
    output_dir = get_checkpoint_folder()
    physical_devices = tf.config.experimental.list_physical_devices("GPU")
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    anchors, anchor_masks, model = load_model_with_ancors(
        num_classes=int(args.new_model_class_count), use_tiny=args.use_tiny
    )

    train_dataset = load_data(
        tfrecord_filepath=args.tfrecord_train,
        anchors=anchors,
        anchor_masks=anchor_masks,
        batch_size=args.batch_size,
    )
    train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    val_dataset = load_data(
        tfrecord_filepath=args.tfrecord_train,
        anchors=anchors,
        anchor_masks=anchor_masks,
        batch_size=args.batch_size,
    )

    transfer_darknet_layer(
        model=model, transfer_weights_path=args.weights, use_tiny=args.use_tiny
    )

    optimizer = tf.keras.optimizers.Adam(lr=args.learning_rate)
    loss = [
        yolo_loss(anchors[mask], num_classes=args.new_model_class_count)
        for mask in anchor_masks
    ]

    model.compile(optimizer=optimizer, loss=loss, run_eagerly=False)

    callbacks = [
        ReduceLROnPlateau(verbose=1),
        EarlyStopping(patience=3, verbose=1),
        ModelCheckpoint(
            output_dir + "/yolov3_train_{epoch}.tf", verbose=1, save_weights_only=True
        ),
        TensorBoard(log_dir="logs"),
    ]

    model.fit(
        train_dataset,
        epochs=args.epochs,
        callbacks=callbacks,
        validation_data=val_dataset,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="For fine tuning yolov3 object detection against new object classes"
    )

    parser.add_argument("-tfrecord-train", default="", help="path to training dataset")
    parser.add_argument("-tfrecord-test", default="", help="path to testing dataset")
    parser.add_argument("--use-tiny", default=False, help="yolov3 or yolov3-tiny")
    parser.add_argument(
        "--weights", default="./checkpoints/yolov3.tf", help="path to weights file"
    )
    parser.add_argument("--epochs", default=2, help="number of epochs")
    parser.add_argument("--batch-size", default=8, help="batch size")
    parser.add_argument("--learning-rate", default=1e-3, help="learning rate")
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
