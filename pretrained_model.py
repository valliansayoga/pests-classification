import tensorflow as tf

def output_layer(inputs):
    x = tf.keras.layers.Flatten()(inputs)
    x = tf.keras.layers.Dense(512, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    outputs = tf.keras.layers.Dense(12, activation="softmax")(x)
    
    return outputs

def mobilenet(height, width, trainable=False):
    pretrained = tf.keras.applications.mobilenet.MobileNet(
        input_shape=(height, width, 3),
        alpha=1.0,
        depth_multiplier=1,
        dropout=0.001,
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

