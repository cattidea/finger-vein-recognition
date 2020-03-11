import tensorflow as tf


class CustomLayer(tf.keras.layers.Layer):

    def __init__(self, **kw):
        self.custom_config = kw
        super().__init__()

    def get_config(self):
        return self.custom_config

def get_custom_objects():
    custom_objects = dict()
    for cls in CustomLayer.__subclasses__():
        custom_objects[cls.__name__] = cls
    return custom_objects

class IdentityLayer(CustomLayer):

    def __init__(self):
        super().__init__()

    def call(self, inputs):
        return inputs
