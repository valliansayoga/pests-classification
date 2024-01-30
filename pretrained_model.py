import tensorflow as tf
from functools import partial


def output_layer(inputs):
    x = tf.keras.layers.Flatten()(inputs)
    x = tf.keras.layers.Dense(512, activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(512, activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(12, activation="softmax")(x)
    
    return outputs

def load_pretrained_model(name, height, width, trainable=False):
    pretrained = name(
        input_shape=(height, width, 3),
        include_top=False,
        weights='imagenet',
        classes=12,
        classifier_activation='softmax',
    )
    for layer in pretrained.layers:
        layer.trainable = trainable

    inputs = pretrained.input
    outputs = output_layer(pretrained.output)
    
    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs
    )
    model.summary()
    return model

mobilenet = partial(load_pretrained_model,
                    name=tf.keras.applications.mobilenet.MobileNet)

vgg19 = partial(load_pretrained_model,
                name=tf.keras.applications.vgg19.VGG19)

inceptionv3 = partial(load_pretrained_model,
                      name=tf.keras.applications.inception_v3.InceptionV3)

