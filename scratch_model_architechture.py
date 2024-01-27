import tensorflow as tf
def model(height, width):
    models = tf.keras.models.Sequential(
        [
            tf.keras.layers.Conv2D(
                32, (3, 3), input_shape=(height, width, 3), activation="relu"
            ),
            tf.keras.layers.MaxPool2D(2, 2),
            tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
            tf.keras.layers.MaxPool2D(2, 2),
            tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
            tf.keras.layers.MaxPool2D(2, 2),
            tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
            tf.keras.layers.MaxPool2D(2, 2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation="relu"),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.Dense(12, activation="softmax"),
        ]
    )
    models.summary()
    return models