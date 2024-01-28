from tensorflow import keras
from PIL import Image
from pathlib import Path
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import scratch_model_architechture as sma
import pretrained_model as pm

def training_generator(path, height, width, seed, batch_size):
    train_data = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=0.25,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        rescale=1.0 / 255,
    )
    train_gen = train_data.flow_from_directory(
        path,
        target_size=(height, width),
        class_mode="sparse",
        batch_size=batch_size,
        seed=seed,
    )
    return train_gen


def non_training_generator(path, height, width, seed, batch_size):
    data = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1.0 / 255,
    )
    gen = data.flow_from_directory(
        path,
        target_size=(height, width),
        batch_size=batch_size,
        class_mode="sparse",
        shuffle=True,
        seed=seed,
    )
    return gen


if __name__ == "__main__":
    DATA_PATH = Path() / "Data" / "Split Data"
    SEED = 0
    WIDTH = 128
    HEIGHT = 128
    TRAIN = DATA_PATH / "train"
    VAL = DATA_PATH / "val"
    TEST = DATA_PATH / "test"
    BATCH_SIZE = 64
    EPOCHS = 100

    os.environ["TF_DETERMINISTIC_OPS"] = str(SEED)
    os.environ["TF_CUDNN_DETERMINISTIC"] = str(SEED)
    tf.config.threading.set_inter_op_parallelism_threads(SEED)
    tf.config.threading.set_intra_op_parallelism_threads(SEED)

    print("Generating data...")
    train_gen = training_generator(
        TRAIN, seed=SEED, height=HEIGHT, width=WIDTH, batch_size=BATCH_SIZE
    )
    val_gen = non_training_generator(
        VAL, seed=SEED, height=HEIGHT, width=WIDTH, batch_size=BATCH_SIZE
    )
    test_gen = non_training_generator(
        TEST, seed=SEED, height=HEIGHT, width=WIDTH, batch_size=BATCH_SIZE
    )
    print()
    
    # # Scratch Model
    # m = sma.model(HEIGHT, WIDTH)
    
    # Pretrained Model
    m = pm.mobilenet(HEIGHT, WIDTH)
    
    m.compile(
        optimizer=tf.keras.optimizers.SGD(
            learning_rate=0.01,
            momentum=0.9,
        ),
        metrics=["accuracy"],
        loss=["sparse_categorical_crossentropy"],
    )
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            "Model/Checkpoint", monitor="val_accuracy", save_best_only=True
        ),
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=10),
        tf.keras.callbacks.TensorBoard("Logs"),
    ]
    m.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        workers=4,
        callbacks=callbacks,
    )
