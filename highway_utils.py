import tensorflow as tf
from nn_utils import linear
from cnn_utils import conv

def highway_layer(arg, bias, bias_start=0.0, scope=None, wd=0.0, input_keep_prob=1.0, is_train=None):
    with tf.variable_scope(scope or "highway_layer"):
        d = arg.get_shape()[-1]  # embedding dim
        trans = linear([arg], d, bias, bias_start=bias_start, scope='trans', wd=wd, input_keep_prob=input_keep_prob,
                       is_train=is_train)
        trans = tf.nn.relu(trans)
        gate = linear([arg], d, bias, bias_start=bias_start, scope='gate', wd=wd, input_keep_prob=input_keep_prob,
                      is_train=is_train)
        gate = tf.nn.sigmoid(gate)
        out = gate * trans + (1 - gate) * arg
        return out


def highway_network(arg, num_layers, bias, bias_start=0.0, scope=None, wd=0.0, input_keep_prob=1.0, is_train=None):
    with tf.variable_scope(scope or "highway_network"):
        prev = arg
        cur = None
        for layer_idx in range(num_layers):
            cur = highway_layer(prev, bias, bias_start=bias_start, scope="layer_{}".format(layer_idx), wd=wd,
                                input_keep_prob=input_keep_prob, is_train=is_train)
            prev = cur
        return cur

def highway(x, size=None, activation=None,
            num_layers=2, scope="highway", dropout=0.0, reuse=None):
    with tf.variable_scope(scope, reuse):
        if size is None:
            size = x.shape.as_list()[-1]
        else:
            x = conv(x, size, name="input_projection", reuse=reuse)
        for i in range(num_layers):
            T = conv(x, size, bias=True, activation=tf.sigmoid,
                     name="gate_%d" % i, reuse=reuse)
            H = conv(x, size, bias=True, activation=activation,
                     name="activation_%d" % i, reuse=reuse)
            H = tf.nn.dropout(H, 1.0 - dropout)
            x = H * T + x * (1.0 - T)
        return x