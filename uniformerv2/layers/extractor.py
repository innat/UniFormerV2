
import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras import layers

from layers import TFDropPath
from layers import TFQuickGELU
from layers import TFMultiheadAttention

class TFExtractor(keras.Model):
    def __init__(
        self, 
        d_model, 
        n_head, 
        attn_mask=None,
        mlp_factor=4.0, 
        dropout=0.0, 
        drop_path=0.0,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.drop_path = TFDropPath(drop_path) if drop_path > 0. else layers.Identity()
        self.attn = TFMultiheadAttention(num_heads=n_head, key_dim=d_model)
        self.ln_1 = layers.LayerNormalization(axis=-1, epsilon=1e-05)
        d_mlp = round(mlp_factor * d_model)
        self.mlp = keras.Sequential(
            [
                layers.Dense(d_mlp),
                TFQuickGELU(),
                layers.Dropout(dropout),
                layers.Dense(d_model)
            ], name='mlp'
        )
        self.ln_2 = layers.LayerNormalization(axis=-1, epsilon=1e-05)
        self.ln_3 = layers.LayerNormalization(axis=-1, epsilon=1e-05)
        self.attn_mask = attn_mask

    def attention(self, x, y):
        return self.attn(query=x, key=y, value=y, attention_mask=self.attn_mask)

    def call(self, x, y, training=None):
        x = x + self.drop_path(self.attention(self.ln_1(x), self.ln_3(y)), training=training)
        x = x + self.drop_path(self.mlp(self.ln_2(x)), training=training)
        return x