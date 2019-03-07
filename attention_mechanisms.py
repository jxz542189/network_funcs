"""
Copyright 2017 Neural Networks and Deep Learning lab, MIPT

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer as xav
from .dropout_utils import dropout_layer
import csoftmax_attention
from log import get_logger
from tensorflow.python.ops import nn_ops


log = get_logger(__name__)

def calcuate_attention(in_value_1, in_value_2, feature_dim1, feature_dim2, scope_name='att',
                       att_type='symmetric', att_dim=20, remove_diagnoal=False, mask1=None, mask2=None, is_training=False, dropout_rate=0.2):
    input_shape = tf.shape(in_value_1)
    batch_size = input_shape[0]
    len_1 = input_shape[1]
    len_2 = tf.shape(in_value_2)[1]

    in_value_1 = dropout_layer(in_value_1, dropout_rate, is_training=is_training)
    in_value_2 = dropout_layer(in_value_2, dropout_rate, is_training=is_training)
    with tf.variable_scope(scope_name):
        # calculate attention ==> a: [batch_size, len_1, len_2]
        atten_w1 = tf.get_variable("atten_w1", [feature_dim1, att_dim], dtype=tf.float32)
        if feature_dim1 == feature_dim2: atten_w2 = atten_w1
        else: atten_w2 = tf.get_variable("atten_w2", [feature_dim2, att_dim], dtype=tf.float32)
        atten_value_1 = tf.matmul(tf.reshape(in_value_1, [batch_size * len_1, feature_dim1]), atten_w1)  # [batch_size*len_1, feature_dim]
        atten_value_1 = tf.reshape(atten_value_1, [batch_size, len_1, att_dim])
        atten_value_2 = tf.matmul(tf.reshape(in_value_2, [batch_size * len_2, feature_dim2]), atten_w2)  # [batch_size*len_2, feature_dim]
        atten_value_2 = tf.reshape(atten_value_2, [batch_size, len_2, att_dim])


        if att_type == 'additive':
            atten_b = tf.get_variable("atten_b", [att_dim], dtype=tf.float32)
            atten_v = tf.get_variable("atten_v", [1, att_dim], dtype=tf.float32)
            atten_value_1 = tf.expand_dims(atten_value_1, axis=2, name="atten_value_1")  # [batch_size, len_1, 'x', feature_dim]
            atten_value_2 = tf.expand_dims(atten_value_2, axis=1, name="atten_value_2")  # [batch_size, 'x', len_2, feature_dim]
            atten_value = atten_value_1 + atten_value_2  # + tf.expand_dims(tf.expand_dims(tf.expand_dims(atten_b, axis=0), axis=0), axis=0)
            atten_value = nn_ops.bias_add(atten_value, atten_b)
            atten_value = tf.tanh(atten_value)  # [batch_size, len_1, len_2, feature_dim]
            atten_value = tf.reshape(atten_value, [-1, att_dim]) * atten_v  # tf.expand_dims(atten_v, axis=0) # [batch_size*len_1*len_2, feature_dim]
            atten_value = tf.reduce_sum(atten_value, axis=-1)
            atten_value = tf.reshape(atten_value, [batch_size, len_1, len_2])
        else:
            atten_value_1 = tf.tanh(atten_value_1)
            # atten_value_1 = tf.nn.relu(atten_value_1)
            atten_value_2 = tf.tanh(atten_value_2)
            # atten_value_2 = tf.nn.relu(atten_value_2)
            diagnoal_params = tf.get_variable("diagnoal_params", [1, 1, att_dim], dtype=tf.float32)
            atten_value_1 = atten_value_1 * diagnoal_params
            atten_value = tf.matmul(atten_value_1, atten_value_2, transpose_b=True) # [batch_size, len_1, len_2]

        # normalize
        if remove_diagnoal:
            diagnoal = tf.ones([len_1], tf.float32)  # [len1]
            diagnoal = 1.0 - tf.diag(diagnoal)  # [len1, len1]
            diagnoal = tf.expand_dims(diagnoal, axis=0)  # ['x', len1, len1]
            atten_value = atten_value * diagnoal
        if mask1 is not None: atten_value = tf.multiply(atten_value, tf.expand_dims(mask1, axis=-1))
        if mask2 is not None: atten_value = tf.multiply(atten_value, tf.expand_dims(mask2, axis=1))
        atten_value = tf.nn.softmax(atten_value, name='atten_value')  # [batch_size, len_1, len_2]
        if remove_diagnoal: atten_value = atten_value * diagnoal
        if mask1 is not None: atten_value = tf.multiply(atten_value, tf.expand_dims(mask1, axis=-1))
        if mask2 is not None: atten_value = tf.multiply(atten_value, tf.expand_dims(mask2, axis=1))

    return atten_value


def general_attention(key, context, hidden_size, projected_align=False):
    """ It is a implementation of the Luong et al. attention mechanism with general score. Based on the paper:
        https://arxiv.org/abs/1508.04025 "Effective Approaches to Attention-based Neural Machine Translation"
    Args:
        key: A tensorflow tensor with dimensionality [None, None, key_size]
        context: A tensorflow tensor with dimensionality [None, None, max_num_tokens, token_size]
        hidden_size: Number of units in hidden representation
        projected_align: Using bidirectional lstm for hidden representation of context.
        If true, beetween input and attention mechanism insert layer of bidirectional lstm with dimensionality [hidden_size].
        If false, bidirectional lstm is not used.
    Returns:
        output: Tensor at the output with dimensionality [None, None, hidden_size]
    """

    if hidden_size % 2 != 0:
        raise ValueError("hidden size must be dividable by two")
    batch_size = tf.shape(context)[0]
    max_num_tokens, token_size = context.get_shape().as_list()[-2:]
    r_context = tf.reshape(context, shape=[-1, max_num_tokens, token_size])

    # projected_key: [None, None, hidden_size]
    projected_key = \
        tf.layers.dense(key, hidden_size, kernel_initializer=xav())
    r_projected_key = tf.reshape(projected_key, shape=[-1, hidden_size, 1])

    lstm_fw_cell = tf.nn.rnn_cell.LSTMCell(hidden_size//2)
    lstm_bw_cell = tf.nn.rnn_cell.LSTMCell(hidden_size//2)
    (output_fw, output_bw), states = \
        tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_fw_cell,
                                        cell_bw=lstm_bw_cell,
                                        inputs=r_context,
                                        dtype=tf.float32)
    # bilstm_output: [-1, max_num_tokens, hidden_size]
    bilstm_output = tf.concat([output_fw, output_bw], -1)

    attn = tf.nn.softmax(tf.matmul(bilstm_output, r_projected_key), dim=1)

    if projected_align:
        log.info("Using projected attention alignment")
        t_context = tf.transpose(bilstm_output, [0, 2, 1])
        output = tf.reshape(tf.matmul(t_context, attn),
                            shape=[batch_size, -1, hidden_size])
    else:
        log.info("Using without projected attention alignment")
        t_context = tf.transpose(r_context, [0, 2, 1])
        output = tf.reshape(tf.matmul(t_context, attn),
                            shape=[batch_size, -1, token_size])
    return output


def light_general_attention(key, context, hidden_size, projected_align=False):
    """ It is a implementation of the Luong et al. attention mechanism with general score. Based on the paper:
        https://arxiv.org/abs/1508.04025 "Effective Approaches to Attention-based Neural Machine Translation"
    Args:
        key: A tensorflow tensor with dimensionality [None, None, key_size]
        context: A tensorflow tensor with dimensionality [None, None, max_num_tokens, token_size]
        hidden_size: Number of units in hidden representation
        projected_align: Using dense layer for hidden representation of context.
        If true, between input and attention mechanism insert a dense layer with dimensionality [hidden_size].
        If false, a dense layer is not used.
    Returns:
        output: Tensor at the output with dimensionality [None, None, hidden_size]
    """
    batch_size = tf.shape(context)[0]
    max_num_tokens, token_size = context.get_shape().as_list()[-2:]
    r_context = tf.reshape(context, shape=[-1, max_num_tokens, token_size])

    # projected_key: [None, None, hidden_size]
    projected_key = tf.layers.dense(key, hidden_size, kernel_initializer=xav())
    r_projected_key = tf.reshape(projected_key, shape=[-1, hidden_size, 1])

    # projected context: [None, None, hidden_size]
    projected_context = \
        tf.layers.dense(r_context, hidden_size, kernel_initializer=xav())

    attn = tf.nn.softmax(tf.matmul(projected_context, r_projected_key), dim=1)

    if projected_align:
        log.info("Using projected attention alignment")
        t_context = tf.transpose(projected_context, [0, 2, 1])
        output = tf.reshape(tf.matmul(t_context, attn),
                            shape=[batch_size, -1, hidden_size])
    else:
        log.info("Using without projected attention alignment")
        t_context = tf.transpose(r_context, [0, 2, 1])
        output = tf.reshape(tf.matmul(t_context, attn),
                            shape=[batch_size, -1, token_size])
    return output


def cs_general_attention(key, context, hidden_size, depth, projected_align=False):
    """ It is a implementation of the Luong et al. attention mechanism with general score and the constrained softmax (csoftmax).
        Based on the papers:
        https://arxiv.org/abs/1508.04025 "Effective Approaches to Attention-based Neural Machine Translation"
        https://andre-martins.github.io/docs/emnlp2017_final.pdf "Learning What's Easy: Fully Differentiable Neural Easy-First Taggers"
    Args:
        key: A tensorflow tensor with dimensionality [None, None, key_size]
        context: A tensorflow tensor with dimensionality [None, None, max_num_tokens, token_size]
        hidden_size: Number of units in hidden representation
        depth: Number of csoftmax usages
        projected_align: Using bidirectional lstm for hidden representation of context.
        If true, beetween input and attention mechanism insert layer of bidirectional lstm with dimensionality [hidden_size].
        If false, bidirectional lstm is not used.
    Returns:
        output: Tensor at the output with dimensionality [None, None, depth * hidden_size]
    """
    if hidden_size % 2 != 0:
        raise ValueError("hidden size must be dividable by two")
    key_size = tf.shape(key)[-1]
    batch_size = tf.shape(context)[0]
    max_num_tokens, token_size = context.get_shape().as_list()[-2:]
    r_context = tf.reshape(context, shape=[-1, max_num_tokens, token_size])
    # projected_context: [None, max_num_tokens, token_size]
    projected_context = tf.layers.dense(r_context, token_size,
                                        kernel_initializer=xav(),
                                        name='projected_context')

    lstm_fw_cell = tf.nn.rnn_cell.LSTMCell(hidden_size//2)
    lstm_bw_cell = tf.nn.rnn_cell.LSTMCell(hidden_size//2)
    (output_fw, output_bw), states = \
        tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_fw_cell,
                                        cell_bw=lstm_bw_cell,
                                        inputs=projected_context,
                                        dtype=tf.float32)
    # bilstm_output: [-1, max_num_tokens, hidden_size]
    bilstm_output = tf.concat([output_fw, output_bw], -1)
    h_state_for_sketch = bilstm_output

    if projected_align:
        log.info("Using projected attention alignment")
        h_state_for_attn_alignment = bilstm_output
        aligned_h_state = csoftmax_attention.attention_gen_block(
            h_state_for_sketch, h_state_for_attn_alignment, key, depth)
        output = \
            tf.reshape(aligned_h_state, shape=[batch_size, -1, depth * hidden_size])
    else:
        log.info("Using without projected attention alignment")
        h_state_for_attn_alignment = projected_context
        aligned_h_state = csoftmax_attention.attention_gen_block(
            h_state_for_sketch, h_state_for_attn_alignment, key, depth)
        output = \
            tf.reshape(aligned_h_state, shape=[batch_size, -1, depth * token_size])
    return output


def bahdanau_attention(key, context, hidden_size, projected_align=False):
    """ It is a implementation of the Bahdanau et al. attention mechanism. Based on the paper:
        https://arxiv.org/abs/1409.0473 "Neural Machine Translation by Jointly Learning to Align and Translate"
    Args:
        key: A tensorflow tensor with dimensionality [None, None, key_size]
        context: A tensorflow tensor with dimensionality [None, None, max_num_tokens, token_size]
        hidden_size: Number of units in hidden representation
        projected_align: Using bidirectional lstm for hidden representation of context.
        If true, beetween input and attention mechanism insert layer of bidirectional lstm with dimensionality [hidden_size].
        If false, bidirectional lstm is not used.
    Returns:
        output: Tensor at the output with dimensionality [None, None, hidden_size]
    """
    if hidden_size % 2 != 0:
        raise ValueError("hidden size must be dividable by two")
    batch_size = tf.shape(context)[0]
    max_num_tokens, token_size = context.get_shape().as_list()[-2:]
    r_context = tf.reshape(context, shape=[-1, max_num_tokens, token_size])

    # projected_key: [None, None, hidden_size]
    projected_key = tf.layers.dense(key, hidden_size, kernel_initializer=xav())
    r_projected_key = \
        tf.tile(tf.reshape(projected_key, shape=[-1, 1, hidden_size]),
                [1, max_num_tokens, 1])

    lstm_fw_cell = tf.nn.rnn_cell.LSTMCell(hidden_size//2)
    lstm_bw_cell = tf.nn.rnn_cell.LSTMCell(hidden_size//2)
    (output_fw, output_bw), states = \
        tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_fw_cell,
                                        cell_bw=lstm_bw_cell,
                                        inputs=r_context,
                                        dtype=tf.float32)

    # bilstm_output: [-1,self.max_num_tokens,_n_hidden]
    bilstm_output = tf.concat([output_fw, output_bw], -1)
    concat_h_state = tf.concat([r_projected_key, output_fw, output_bw], -1)
    projected_state = \
        tf.layers.dense(concat_h_state, hidden_size, use_bias=False,
                        kernel_initializer=xav())
    score = \
        tf.layers.dense(tf.tanh(projected_state), units=1, use_bias=False,
                        kernel_initializer=xav())

    attn = tf.nn.softmax(score, dim=1)

    if projected_align:
        log.info("Using projected attention alignment")
        t_context = tf.transpose(bilstm_output, [0, 2, 1])
        output = tf.reshape(tf.matmul(t_context, attn),
                            shape=[batch_size, -1, hidden_size])
    else:
        log.info("Using without projected attention alignment")
        t_context = tf.transpose(r_context, [0, 2, 1])
        output = tf.reshape(tf.matmul(t_context, attn),
                            shape=[batch_size, -1, token_size])
    return output


def light_bahdanau_attention(key, context, hidden_size, projected_align=False):
    """ It is a implementation of the Bahdanau et al. attention mechanism. Based on the paper:
        https://arxiv.org/abs/1409.0473 "Neural Machine Translation by Jointly Learning to Align and Translate"
    Args:
        key: A tensorflow tensor with dimensionality [None, None, key_size]
        context: A tensorflow tensor with dimensionality [None, None, max_num_tokens, token_size]
        hidden_size: Number of units in hidden representation
        projected_align: Using dense layer for hidden representation of context.
        If true, between input and attention mechanism insert a dense layer with dimensionality [hidden_size].
        If false, a dense layer is not used.
    Returns:
        output: Tensor at the output with dimensionality [None, None, hidden_size]
    """
    batch_size = tf.shape(context)[0]
    max_num_tokens, token_size = context.get_shape().as_list()[-2:]
    r_context = tf.reshape(context, shape=[-1, max_num_tokens, token_size])

    # projected_key: [None, None, hidden_size]
    projected_key = tf.layers.dense(key, hidden_size, kernel_initializer=xav())
    r_projected_key = \
        tf.tile(tf.reshape(projected_key, shape=[-1, 1, hidden_size]),
                [1, max_num_tokens, 1])

    # projected_context: [None, max_num_tokens, hidden_size]
    projected_context = \
        tf.layers.dense(r_context, hidden_size, kernel_initializer=xav())
    concat_h_state = tf.concat([projected_context, r_projected_key], -1)

    projected_state = \
        tf.layers.dense(concat_h_state, hidden_size, use_bias=False,
                        kernel_initializer=xav())
    score = \
        tf.layers.dense(tf.tanh(projected_state), units=1, use_bias=False,
                        kernel_initializer=xav())

    attn = tf.nn.softmax(score, dim=1)

    if projected_align:
        log.info("Using projected attention alignment")
        t_context = tf.transpose(projected_context, [0, 2, 1])
        output = tf.reshape(tf.matmul(t_context, attn),
                            shape=[batch_size, -1, hidden_size])
    else:
        log.info("Using without projected attention alignment")
        t_context = tf.transpose(r_context, [0, 2, 1])
        output = tf.reshape(tf.matmul(t_context, attn),
                            shape=[batch_size, -1, token_size])
    return output


def cs_bahdanau_attention(key, context, hidden_size, depth, projected_align=False):
    """ It is a implementation of the Bahdanau et al. attention mechanism. Based on the papers:
        https://arxiv.org/abs/1409.0473 "Neural Machine Translation by Jointly Learning to Align and Translate"
        https://andre-martins.github.io/docs/emnlp2017_final.pdf "Learning What's Easy: Fully Differentiable Neural Easy-First Taggers"
    Args:
        key: A tensorflow tensor with dimensionality [None, None, key_size]
        context: A tensorflow tensor with dimensionality [None, None, max_num_tokens, token_size]
        hidden_size: Number of units in hidden representation
        depth: Number of csoftmax usages
        projected_align: Using bidirectional lstm for hidden representation of context.
        If true, beetween input and attention mechanism insert layer of bidirectional lstm with dimensionality [hidden_size].
        If false, bidirectional lstm is not used.
    Returns:
        output: Tensor at the output with dimensionality [None, None, depth * hidden_size]
    """
    if hidden_size % 2 != 0:
        raise ValueError("hidden size must be dividable by two")
    batch_size = tf.shape(context)[0]
    max_num_tokens, token_size = context.get_shape().as_list()[-2:]

    r_context = tf.reshape(context, shape=[-1, max_num_tokens, token_size])
    # projected context: [None, max_num_tokens, token_size]
    projected_context = tf.layers.dense(r_context, token_size,
                                        kernel_initializer=xav(),
                                        name='projected_context')

    # projected_key: [None, None, hidden_size]
    projected_key = tf.layers.dense(key, hidden_size, kernel_initializer=xav(),
                                    name='projected_key')
    r_projected_key = \
        tf.tile(tf.reshape(projected_key, shape=[-1, 1, hidden_size]),
                [1, max_num_tokens, 1])

    lstm_fw_cell = tf.nn.rnn_cell.LSTMCell(hidden_size//2)
    lstm_bw_cell = tf.nn.rnn_cell.LSTMCell(hidden_size//2)
    (output_fw, output_bw), states = \
        tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_fw_cell,
                                        cell_bw=lstm_bw_cell,
                                        inputs=projected_context,
                                        dtype=tf.float32)

    # bilstm_output: [-1, max_num_tokens, hidden_size]
    bilstm_output = tf.concat([output_fw, output_bw], -1)
    concat_h_state = tf.concat([r_projected_key, output_fw, output_bw], -1)

    if projected_align:
        log.info("Using projected attention alignment")
        h_state_for_attn_alignment = bilstm_output
        aligned_h_state = csoftmax_attention.attention_bah_block(
            concat_h_state, h_state_for_attn_alignment, depth)
        output = \
            tf.reshape(aligned_h_state, shape=[batch_size, -1, depth * hidden_size])
    else:
        log.info("Using without projected attention alignment")
        h_state_for_attn_alignment = projected_context
        aligned_h_state = csoftmax_attention.attention_bah_block(
            concat_h_state, h_state_for_attn_alignment, depth)
        output = \
            tf.reshape(aligned_h_state, shape=[batch_size, -1, depth * token_size])
    return output
