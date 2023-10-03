
import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras import layers

from layers import TFExtractor
from blocks import TFResidualAttentionBlock


class TFTransformer(keras.Model):
    def __init__(
        self, 
        width, 
        resblocks_layers, 
        heads, 
        attn_mask=None, 
        backbone_drop_path_rate=0., 
        t_size=8, 
        dw_reduction=2,
        no_lmhra=False, 
        double_lmhra=True,
        return_list=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        n_layers=12, 
        n_dim=768, 
        n_head=12, 
        mlp_factor=4.0, 
        drop_path_rate=0.,
        mlp_dropout=[0.5]*12, 
        cls_dropout=0.5, 
        num_classes=400,
        frozen=False,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.T = t_size
        self.return_list = return_list
        
        # backbone
        b_dpr = tf.linspace(0., backbone_drop_path_rate, resblocks_layers).numpy().tolist() 
        self.resblocks = [
            TFResidualAttentionBlock(
                width, heads, attn_mask, 
                drop_path=b_dpr[i],
                dw_reduction=dw_reduction,
                no_lmhra=no_lmhra,
                double_lmhra=double_lmhra,
                name=f'TFResidualAttentionBlock{i+1}'
            ) for i in range(resblocks_layers)
        ]
        
        # global block
        assert n_layers == len(return_list)
        self.temporal_cls_token = self.add_weight(
            'temporal_cls_token',
            shape=(1, 1, n_dim), 
            initializer='zeros'
        )
        self.dpe = [
            layers.Conv3D(
                n_dim, kernel_size=3, strides=1, padding='same', groups=n_dim
            ) for _ in range(n_layers)
        ]
        
        dpr = tf.linspace(0., drop_path_rate, n_layers).numpy().tolist() 
        self.dec = [
            TFExtractor(
                n_dim,
                n_head, 
                mlp_factor=mlp_factor, 
                dropout=mlp_dropout[i], 
                drop_path=dpr[i],
                name=f'TFExtractor{i+1}'
            ) for i in range(n_layers)
        ]
        
        # projection
        self.proj = keras.Sequential(
            [
                layers.LayerNormalization(axis=-1, epsilon=1e-05),
                layers.Dropout(cls_dropout),
                layers.Dense(num_classes)
            ], name="proj"
        )
        
        self.frozen = frozen
        if not self.frozen:
            self.balance = self.add_weight('balance', shape=(n_dim,), initializer='zeros')

        
    def call(self, x, training=None):
        T_down = self.T
        L, NT, C = tf.unstack(tf.shape(x))
        N = NT // T_down
        H = W = tf.cast(tf.math.sqrt(tf.cast((L-1), dtype=tf.float32)), dtype=tf.int32)
        cls_token = tf.tile(self.temporal_cls_token, [1, N, 1])

        j = -1
        for i, resblock in enumerate(self.resblocks):
            x = resblock(x, T_down, training=training)
            
            if i in self.return_list:
                j += 1
                tmp_x = tf.reshape(x, [L, N, T_down, C])

                # dpe 
                _, tmp_feats = tmp_x[:1], tmp_x[1:]
                tmp_feats = tf.transpose(tmp_feats, [1, 2, 0, 3])
                tmp_feats = tf.reshape(tmp_feats, [N, T_down, H, W, C])
                tmp_feats = self.dpe[j](tmp_feats)
                tmp_feats = tf.reshape(tmp_feats, [N, T_down, L - 1, C])
                tmp_feats = tf.transpose(tmp_feats, [2, 0, 1, 3])
                tmp_x = tf.concat(
                    [tmp_x[:1], tmp_x[1:] + tmp_feats], axis=0
                )

                # global block
                tmp_x = tf.transpose(tmp_x, [2, 0, 1, 3])
                tmp_x = tf.reshape(tmp_x, [T_down * L, N, C])
                cls_token = self.dec[j](cls_token, tmp_x)
                

        if self.frozen:
            return self.proj(cls_token[0, :, :])
        else:
            weight = tf.nn.sigmoid(self.balance)
            residual = tf.reduce_mean(tf.reshape(x, [L, N, T_down, C])[0], axis=1)
            return self.proj((1 - weight) * cls_token[0, :, :] + weight * residual)