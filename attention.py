import numpy as np
import tensorflow as tf
from keras import backend as K
from tensorflow.keras.initializers import Ones, Zeros
from tensorflow.keras.layers import Activation, Add, Dense, Dropout, Lambda, Layer


class LayerNormalization(Layer):

    def __init__(self, eps=1e-6, **kwargs):

        self.eps = eps

        super(LayerNormalization, self).__init__(**kwargs)

    def build(self, input_shape):

        self.gamma = self.add_weight(
            name="gamma", shape=input_shape[-1:], initializer=Ones(), trainable=True
        )

        self.beta = self.add_weight(
            name="beta", shape=input_shape[-1:], initializer=Zeros(), trainable=True
        )

        super(LayerNormalization, self).build(input_shape)

    def call(self, x):

        mean = K.mean(x, axis=-1, keepdims=True)

        std = K.std(x, axis=-1, keepdims=True)

        return self.gamma * (x - mean) / (std + self.eps) + self.beta

    def compute_output_shape(self, input_shape):

        return input_shape


class ScaledDotProductAttention:

    def __init__(self, d_model, attn_dropout=0.1):

        self.temper = np.sqrt(d_model)

        self.dropout = Dropout(attn_dropout)

    def __call__(self, q, k, v, mask):

        attn = Lambda(lambda x: K.batch_dot(x[0], x[1], axes=[2, 2]) / self.temper)(
            [q, k]
        )

        if mask is not None:

            mmask = Lambda(lambda x: (-1e10) * (1 - x))(mask)

            attn = Add()([attn, mmask])

        attn = Activation("softmax")(attn)

        attn = self.dropout(attn)

        output = Lambda(lambda x: K.batch_dot(x[0], x[1]))([attn, v])

        return output, attn


class MultiHeadAttention:

    def __init__(self, n_head, d_model, d_k, d_v, dropout, use_norm=True):

        self.n_head = n_head

        self.d_k = d_k

        self.d_v = d_v

        self.dropout = dropout

        self.qs_layer = Dense(n_head * d_k, use_bias=False)

        self.ks_layer = Dense(n_head * d_k, use_bias=False)

        self.vs_layer = Dense(n_head * d_v, use_bias=False)

        self.attention = ScaledDotProductAttention(d_model)

        self.layer_norm = LayerNormalization() if use_norm else None

        self.w_o = TimeDistributed(Dense(d_model))

    def __call__(self, q, k, v, mask=None):

        d_k, d_v = self.d_k, self.d_v

        n_head = self.n_head

        qs = self.qs_layer(q)  # [batch_size, len_q, n_head*d_k]

        ks = self.ks_layer(k)

        vs = self.vs_layer(v)

        def reshape1(x):

            s = tf.shape(x)  # [batch_size, len_q, n_head * d_k]

            x = tf.reshape(x, [s[0], s[1], n_head, d_k])

            x = tf.transpose(x, [2, 0, 1, 3])

            x = tf.reshape(x, [-1, s[1], d_k])  # [n_head * batch_size, len_q, d_k]

            return x

        qs = Lambda(reshape1)(qs)

        ks = Lambda(reshape1)(ks)

        vs = Lambda(reshape1)(vs)

        if mask is not None:

            mask = Lambda(lambda x: K.repeat_elements(x, n_head, 0))(mask)

        head, attn = self.attention(qs, ks, vs, mask=mask)

        def reshape2(x):

            s = tf.shape(x)  # [n_head * batch_size, len_v, d_v]

            x = tf.reshape(x, [n_head, -1, s[1], s[2]])

            x = tf.transpose(x, [1, 2, 0, 3])

            x = tf.reshape(
                x, [-1, s[1], n_head * d_v]
            )  # [batch_size, len_v, n_head * d_v]

            return x

        head = Lambda(reshape2)(head)

        outputs = self.w_o(head)

        outputs = Dropout(self.dropout)(outputs)

        if not self.layer_norm:

            return outputs, attn

        return self.layer_norm(outputs), attn
