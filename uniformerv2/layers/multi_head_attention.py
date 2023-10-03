import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras import layers

class TFMultiheadAttention(keras.Model):
    def __init__(self, num_heads, key_dim, dropout=0.0, **kwargs):
        super().__init__(**kwargs)
        self.key_dim = key_dim
        self.num_heads = num_heads
        self.head_dim = key_dim // num_heads

        assert (
            self.head_dim * num_heads == key_dim
        ), "key_dim size needs to be divisible by num_heads"

        # Create weights for query, key, and value projections
        self.wq = layers.Dense(key_dim)
        self.wk = layers.Dense(key_dim)
        self.wv = layers.Dense(key_dim)
        
        # Output dense layer
        self.fc_out = layers.Dense(key_dim)
        
        # attn dropput
        self.dropout =  layers.Dropout(rate=dropout)

    def transpose_qkv(self, x, T, N):
        x = tf.reshape(x, [T, N, self.num_heads, self.head_dim])
        x = tf.transpose(x, [1, 2, 0, 3])
        return x

    def call(
        self, 
        query, 
        key, 
        value, 
        attention_mask=None, 
        return_attention_scores=False, 
        training=None
    ):
        batch_size = tf.shape(query)[0]
        
        # Linear projections
        query = self.wq(query)
        key = self.wk(key)
        value = self.wv(value)

        # transposing
        Tx, Ty, N = tf.shape(query)[0], tf.shape(key)[0], tf.shape(query)[1]
        query = self.transpose_qkv(query, Tx, N)
        key = self.transpose_qkv(key, Ty, N)
        value = self.transpose_qkv(value, Ty, N)

        # Dot product attention
        matmul_qk = tf.matmul(query, key, transpose_b=True)
        d_k = tf.cast(self.head_dim, dtype=matmul_qk.dtype)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(d_k)

        if attention_mask is not None:
            scaled_attention_logits += (attention_mask * -1e9)

        # Apply softmax
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        attention_weights = self.dropout(attention_weights, training=training)
        
        # matmul between qk and v
        attention_output = tf.matmul(attention_weights, value)
        attention_output = tf.transpose(attention_output, perm=[2, 0, 1, 3])
        attention_output = tf.reshape(attention_output, (batch_size, -1, self.key_dim))
        attention_output = self.fc_out(attention_output)
    
        if return_attention_scores:
            return attention_output, attention_weights
        return attention_output