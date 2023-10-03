
from tensorflow import keras 
from tensorflow.keras import layers

class TFLocalMultiHeadRelationAggregator(keras.Model):
    def __init__(self, d_model, dw_reduction=1.5, pos_kernel_size=3, **kwargs):
        super().__init__(**kwargs)
        
        padding = pos_kernel_size // 2
        re_d_model = int(d_model // dw_reduction)

        self.pos_embed = keras.Sequential([
            layers.BatchNormalization(axis=-1),
            layers.Conv3D(filters=re_d_model, kernel_size=1, strides=1, padding="valid"),
            layers.Conv3D(
                filters=re_d_model, 
                kernel_size=(pos_kernel_size, 1, 1),
                strides=1, 
                padding="same", 
                groups=re_d_model
            ),
            layers.Conv3D(
                filters=d_model, 
                kernel_size=1, 
                strides=1, 
                padding="valid",
                kernel_initializer='zeros', 
                bias_initializer='zeros'
            )
        ])
    
    def call(self, inputs, training=None):
        return self.pos_embed(inputs, training=training)