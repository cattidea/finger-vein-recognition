import tensorflow as tf
import numpy as np
from stn import spatial_transformer_network as transformer
from layers.utils import CustomLayer


class SpatialTransformer(CustomLayer):
    def __init__(self, out_dims=None):
        super().__init__(out_dims=out_dims)
        self.out_dims = out_dims

    def call(self, inputs):

        x, x_loc = inputs
        if self.out_dims is None:
            B, H, W, C = x.shape.as_list()
            self.out_dims = (H, W)
        h_trans = transformer(x, x_loc, self.out_dims)
        return h_trans


class SpatialTransformerLayer(CustomLayer):
    def __init__(self, theta=[[1., 0, 0], [0, 1., 0]], out_dims=None):
        super().__init__(theta=theta, out_dims=out_dims)
        self.out_dims = out_dims
        self.theta = theta

    def build(self, input_shape):

        B, H, W, C = input_shape
        if self.out_dims is None:
            self.out_dims = (H, W)

        theta = np.array(self.theta, dtype=np.float32)
        theta = theta.flatten()

        layers = []
        # num_layers = int(np.floor(np.log2(min(H, W) / 7)))
        num_layers = int(np.floor(np.log2((min(H, W)-1)/7)) + 1)

        for i in range(num_layers):
            layers.append(
                tf.keras.layers.Conv2D(2**min(i+4, 7), (3, 3), strides=2, activation='relu', padding='same'))
        layers.extend([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(6, kernel_initializer=tf.zeros_initializer(),
                                    bias_initializer=tf.constant_initializer(theta))])

        self.localisation_net = tf.keras.Sequential(layers)

    def call(self, inputs):

        x_loc = self.localisation_net(inputs)
        h_trans = transformer(inputs, x_loc, self.out_dims)
        return h_trans

    def get_dense_weights(self):
        return self.localisation_net.layers[-1].weights

    def compute_theta(self, inputs):
        return self.localisation_net(inputs).numpy().reshpe((2, 3))
