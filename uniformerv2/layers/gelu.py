
import tensorflow as tf
from tensorflow.keras import layers

class TFQuickGELU(layers.Layer):
    def call(self, x):
        return x * tf.nn.sigmoid(1.702 * x)