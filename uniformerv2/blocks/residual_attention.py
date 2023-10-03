
import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras import layers

from layers import TFDropPath
from layers import TFQuickGELU
from layers import TFLocalMultiHeadRelationAggregator
from layers import TFMultiheadAttention


class TFResidualAttentionBlock(keras.Model):
    def __init__(
        self, 
        d_model, 
        n_head, 
        attn_mask=None, 
        drop_path=0.0, 
        dw_reduction=1.5, 
        no_lmhra=False, 
        double_lmhra=True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.drop_path = TFDropPath(drop_path) if drop_path > 0. else layers.Identity()
        self.no_lmhra = no_lmhra
        self.double_lmhra = double_lmhra

        if not no_lmhra:
            self.lmhra1 = TFLocalMultiHeadRelationAggregator(
                d_model, dw_reduction=dw_reduction
            )
            if double_lmhra:
                self.lmhra2 = TFLocalMultiHeadRelationAggregator(
                    d_model, dw_reduction=dw_reduction
                )
        
        # spatial
        self.attn = TFMultiheadAttention(num_heads=n_head, key_dim=d_model)
        self.ln_1 = layers.LayerNormalization(axis=-1, epsilon=1e-05)
        self.mlp = keras.Sequential([
            layers.Dense(d_model * 4),
            TFQuickGELU(),
            layers.Dense(d_model)
        ])
        
        self.ln_2 = layers.LayerNormalization(axis=-1, epsilon=1e-05)
        self.attn_mask = attn_mask
    
    def attention(self, x):
        return self.attn(x, x, x, attention_mask=self.attn_mask)
    
    def call(self, x, T=8, training=None):

        # x: [L+1, NT, C]
        if not self.no_lmhra:
            tmp_x = x[1:, :, :]
            L, NT, C = tf.unstack(tf.shape(tmp_x))
            N = NT // T
            H = W = tf.cast(tf.math.sqrt(tf.cast(L, dtype=tf.float32)), dtype=tf.int32)
            tmp_x = tf.reshape(tmp_x, [H, W, N, T, C])
            tmp_x = tf.transpose(tmp_x, [2, 3, 0, 1, 4])
            tmp_x = tmp_x + self.drop_path(self.lmhra1(tmp_x), training=training)
            tmp_x = tf.reshape(tmp_x, [N, T, L, C])
            tmp_x = tf.transpose(tmp_x, [2, 0, 1, 3])
            tmp_x = tf.reshape(tmp_x, [L, NT, C])
            x = tf.concat([x[:1, :, :], tmp_x], axis=0)

        # MHSA
        x = x + self.drop_path(self.attention(self.ln_1(x)), training=training)

        # Local MHRA
        if not self.no_lmhra and self.double_lmhra:
            tmp_x = x[1:, :, :]
            tmp_x = tf.reshape(tmp_x, [H, W, N, T, C])
            tmp_x = tf.transpose(tmp_x, [2, 3, 0, 1, 4])
            tmp_x = tmp_x + self.drop_path(self.lmhra2(tmp_x), training=training)
            tmp_x = tf.reshape(tmp_x, [N, T, L, C])
            tmp_x = tf.transpose(tmp_x, [2, 0, 1, 3])
            tmp_x = tf.reshape(tmp_x, [L, NT, C])
            x = tf.concat([x[:1, :, :], tmp_x], axis=0)

        # FFN
        x = x + self.drop_path(self.mlp(self.ln_2(x)), training=training)

        return x