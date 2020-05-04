import argparse
import tensorflow as tf
import numpy as np
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


def load_model_with_ancors(num_classes, use_tiny):
    network = yolo_v3(
        training=True,
        num_classes=num_classes,
        use_tiny=use_tiny,
        just_model=False,
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


def transfer_darknet_layer(
    model,
    transfer_weights_classes,
    transfer_weights_path,
    use_tiny
):
    model_pretrained = yolo_v3(
        training=True,
        num_classes=transfer_weights_classes,
        use_tiny=use_tiny,
    )
    model_pretrained.load_weights(transfer_weights_path)
    model.get_layer("yolo_darknet").set_weights(
        model_pretrained.get_layer("yolo_darknet").get_weights()
    )
    freeze_all(model.get_layer("yolo_darknet"))


def main(args):
    physical_devices = tf.config.experimental.list_physical_devices("GPU")
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    anchors, anchor_masks, model = load_model_with_ancors(
        num_classes=int(args.weights_num_classes),
        use_tiny=args.use_tiny)

    train_dataset = load_data(
        tfrecord_filepath=args.tfrecord_train,
        anchors=anchors,
        anchor_masks=anchor_masks,
        batch_size=args.batch_size)
    train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    val_dataset = load_data(
        tfrecord_filepath=args.tfrecord_train,
        anchors=anchors,
        anchor_masks=anchor_masks,
        batch_size=args.batch_size)

    transfer_darknet_layer(
        model=model,
        transfer_weights_classes=args.weights_num_classes,
        transfer_weights_path=args.weights,
        use_tiny=args.use_tiny)

    optimizer = tf.keras.optimizers.Adam(lr=args.learning_rate)
    loss = [
        yolo_loss(anchors[mask], num_classes=args.weights_num_classes) for mask in anchor_masks
    ]

    if args.mode == "eager_tf":
        # Eager mode is great for debugging
        # Non eager graph mode is recommended for real training
        avg_loss = tf.keras.metrics.Mean("loss", dtype=tf.float32)
        avg_val_loss = tf.keras.metrics.Mean("val_loss", dtype=tf.float32)

        for epoch in range(1, args.epochs + 1):
            for batch, (images, labels) in enumerate(train_dataset):
                with tf.GradientTape() as tape:
                    outputs = model(images, training=True)
                    regularization_loss = tf.reduce_sum(model.losses)
                    pred_loss = []
                    for output, label, loss_fn in zip(outputs, labels, loss):
                        pred_loss.append(loss_fn(label, output))
                    total_loss = tf.reduce_sum(pred_loss) + regularization_loss

                grads = tape.gradient(total_loss, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))

                LOGGER.info(
                    "{}_train_{}, {}, {}".format(
                        epoch,
                        batch,
                        total_loss.numpy(),
                        list(map(lambda x: np.sum(x.numpy()), pred_loss)),
                    )
                )
                avg_loss.update_state(total_loss)

            for batch, (images, labels) in enumerate(val_dataset):
                outputs = model(images)
                regularization_loss = tf.reduce_sum(model.losses)
                pred_loss = []
                for output, label, loss_fn in zip(outputs, labels, loss):
                    pred_loss.append(loss_fn(label, output))
                total_loss = tf.reduce_sum(pred_loss) + regularization_loss

                LOGGER.info(
                    "{}_val_{}, {}, {}".format(
                        epoch,
                        batch,
                        total_loss.numpy(),
                        list(map(lambda x: np.sum(x.numpy()), pred_loss)),
                    )
                )
                avg_val_loss.update_state(total_loss)

            LOGGER.info(
                "{}, train: {}, val: {}".format(
                    epoch, avg_loss.result().numpy(), avg_val_loss.result().numpy()
                )
            )

            avg_loss.reset_states()
            avg_val_loss.reset_states()
            model.save_weights("checkpoints/yolov3_train_{}.tf".format(epoch))
    else:
        model.compile(
            optimizer=optimizer, loss=loss, run_eagerly=(args.mode == "eager_fit")
        )

        callbacks = [
            ReduceLROnPlateau(verbose=1),
            EarlyStopping(patience=3, verbose=1),
            ModelCheckpoint(
                "checkpoints/yolov3_train_{epoch}.tf", verbose=1, save_weights_only=True
            ),
            TensorBoard(log_dir="logs"),
        ]

        history = model.fit(
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
    parser.add_argument("--weights", default="./checkpoints/yolov3.tf", help="path to weights file")
    parser.add_argument("--mode",
        default="fit",
        choices=["fit", "eager_fit", "eager_tf"],
        help="fit: model.fit, "
        "eager_fit: model.fit(run_eagerly=True), "
        "eager_tf: custom GradientTape",
    )
    parser.add_argument("--epochs", default=2, help="number of epochs")
    parser.add_argument("--batch-size", default=8, help="batch size")
    parser.add_argument("--learning-rate", default=1e-3, help="learning rate")
    parser.add_argument("--weights-num-classes",
        default=80,
        help="specify num class for `weights` file if different, useful in transfer learning with different number of classes",
    )
    args = parser.parse_args()

    args.epochs = int(args.epochs)
    args.batch_size = int(args.batch_size)
    args.learning_rate = float(args.learning_rate)
    args.weights_num_classes = int(args.weights_num_classes)
    main(args)
