
import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras import layers
from tensorflow.keras import initializers

from uniformerv2.blocks import TFTransformer
from .model_configs import MODEL_CONFIGS

class TFVisionTransformer(keras.Model):
    def __init__(
        self, 
        # backbone
        input_resolution, 
        patch_size, 
        width, 
        vit_layers, 
        heads, 
        output_dim, 
        backbone_drop_path_rate=0.,
        t_size=8, 
        kernel_size=3, 
        dw_reduction=1.5,
        temporal_downsample=True,
        no_lmhra=-False, 
        double_lmhra=True,
        # global block
        return_list=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        n_layers=12, 
        n_dim=768, 
        n_head=12, 
        mlp_factor=4.0, 
        drop_path_rate=0.,
        mlp_dropout=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], 
        cls_dropout=0.5, 
        num_classes=400,
        frozen=False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.temporal_downsample = temporal_downsample
        self.padding = (kernel_size - 1) // 2
        
        if temporal_downsample:
            self.conv1 = layers.Conv3D(
                filters=width, 
                kernel_size=(kernel_size, patch_size, patch_size), 
                strides=(2, patch_size, patch_size), 
                padding="valid",
                use_bias=False,
                name='conv1'
            )
            t_size = t_size // 2
        else:
            self.conv1 = layers.Conv3D(
                filters=width, 
                kernel_size=(1, patch_size, patch_size), 
                strides=(1, patch_size, patch_size), 
                padding="valid", 
                use_bias=False,
                name='conv1'
            )
            
        scale = width ** -0.5
        self.class_embedding = self.add_weight(
            "class_embedding",
            shape=(width,), 
            initializer=initializers.RandomNormal(mean=0., stddev=scale), 
            trainable=True
        )
        self.positional_embedding = self.add_weight(
            "positional_embedding",
            shape=((input_resolution // patch_size) ** 2 + 1, width), 
            initializer=initializers.RandomNormal(mean=0., stddev=scale), 
            trainable=True
        )

        self.ln_pre = layers.LayerNormalization(axis=-1, epsilon=1e-05, name='ln_pre')
        self.transformer = TFTransformer(
            width, vit_layers, heads, dw_reduction=dw_reduction, 
            backbone_drop_path_rate=backbone_drop_path_rate, 
            t_size=t_size, no_lmhra=no_lmhra, double_lmhra=double_lmhra,
            return_list=return_list, n_layers=n_layers, n_dim=n_dim, n_head=n_head, 
            mlp_factor=mlp_factor, drop_path_rate=drop_path_rate, mlp_dropout=mlp_dropout, 
            cls_dropout=cls_dropout, num_classes=num_classes,
            frozen=frozen, name="TFTransformer"
        )
    
    def call(self, x, training=None):
        if self.temporal_downsample:
            x = tf.pad(
                x, 
                [[0, 0], [self.padding, self.padding], [0, 0], [0, 0], [0, 0]]
            ) 
        x = self.conv1(x)
        N, T, H, W, C = tf.unstack(tf.shape(x))
        x = tf.reshape(x, [N * T, H * W, C])

        class_embedding = tf.cast(self.class_embedding, dtype=x.dtype)
        x = tf.concat(
            [
                class_embedding + tf.zeros([tf.shape(x)[0], 1, tf.shape(x)[-1]], dtype=x.dtype), 
                x
            ], 
            axis=1
        )
        x = x + tf.cast(self.positional_embedding, dtype=x.dtype)
        x = self.ln_pre(x)
        x = tf.transpose(x, [1, 0, 2])  # L, N, D
        out = self.transformer(x, training=training)
        return out
    
    def build(self, input_shape):
        super().build(input_shape)
        self.build_shape = input_shape[1:]
    
    def build_graph(self):
        x = keras.Input(shape=self.build_shape, name='input_graph')
        return keras.Model(
            inputs=[x], outputs=self.call(x)
        )
    


def UniFormerV2(name='K400_B16_8x224'):
    # get model variants specific config.
    config = MODEL_CONFIGS[name].copy()

    # set general config.
    dw_reduction=1.5
    backbone_drop_path_rate=0.
    mlp_factor=4.0
    drop_path_rate=0.
    cls_dropout=0.5

    # get model size specific config.
    if 'B' in name:
        patch_size=16
        width=768
        heads=12
        n_head=12
        n_dim=768
        output_dim=512
        vit_layers=12
    elif 'L' in name:
        patch_size=14
        width=1024
        heads=16
        n_head=16
        n_dim=1024
        output_dim=768
        vit_layers=24
    else:
        raise ValueError(
            'UniFormerV2 has only base and large variants.'
        )

    model = TFVisionTransformer(
        patch_size=patch_size,
        width=width,
        vit_layers=vit_layers,
        heads=heads,
        output_dim=output_dim,
        n_dim=n_dim, 
        n_head=n_head, 
        mlp_factor=mlp_factor, 
        drop_path_rate=drop_path_rate, 
        dw_reduction=dw_reduction, 
        backbone_drop_path_rate=backbone_drop_path_rate, 
        cls_dropout=cls_dropout,
        **config
    )
    return model
