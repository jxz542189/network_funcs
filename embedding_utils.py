import tensorflow as tf


def embedding_layer(token_indices=None,
                    token_embedding_matrix=None,
                    n_tokens=None,
                    token_embedding_dim=None,
                    name: str = None,
                    trainable=True):
    """ Token embedding layer. Create matrix of for token embeddings.
        Can be initialized with given matrix (for example pre-trained
        with word2ve algorithm

    Args:
        token_indices: token indices tensor of type tf.int32
        token_embedding_matrix: matrix of embeddings with dimensionality
            [n_tokens, embeddings_dimension]
        n_tokens: total number of unique tokens
        token_embedding_dim: dimensionality of embeddings, typical 100..300
        name: embedding matrix name (variable name)
        trainable: whether to set the matrix trainable or not

    Returns:
        embedded_tokens: tf tensor of size [B, T, E], where B - batch size
            T - number of tokens, E - token_embedding_dim
    """
    if token_embedding_matrix is not None:
        tok_mat = token_embedding_matrix
        if trainable:
            Warning('Matrix of embeddings is passed to the embedding_layer, '
                    'possibly there is a pre-trained embedding matrix. '
                    'Embeddings paramenters are set to Trainable!')
    else:
        tok_mat = np.random.randn(n_tokens, token_embedding_dim).astype(np.float32) / np.sqrt(token_embedding_dim)
    tok_emb_mat = tf.Variable(tok_mat, name=name, trainable=trainable)
    embedded_tokens = tf.nn.embedding_lookup(tok_emb_mat, token_indices)
    return embedded_tokens


def character_embedding_network(char_placeholder: tf.Tensor,
                                n_characters: int =  None,
                                emb_mat: np.array = None,
                                char_embedding_dim: int = None,
                                filter_widths=(3, 4, 5, 7),
                                highway_on_top=False):
    """ Characters to vector. Every sequence of characters (token)
        is embedded to vector space with dimensionality char_embedding_dim
        Convolution plus max_pooling is used to obtain vector representations
        of words.

    Args:
        char_placeholder: placeholder of int32 type with dimensionality [B, T, C]
            B - batch size (can be None)
            T - Number of tokens (can be None)
            C - number of characters (can be None)
        n_characters: total number of unique characters
        emb_mat: if n_characters is not provided the emb_mat should be provided
            it is a numpy array with dimensions [V, E], where V - vocabulary size
            and E - embeddings dimension
        char_embedding_dim: dimensionality of characters embeddings
        filter_widths: array of width of kernel in convolutional embedding network
            used in parallel

    Returns:
        embeddings: tf.Tensor with dimensionality [B, T, F],
            where F is dimensionality of embeddings
    """
    if emb_mat is None:
        emb_mat = np.random.randn(n_characters, char_embedding_dim).astype(np.float32) / np.sqrt(char_embedding_dim)
    else:
        char_embedding_dim = emb_mat.shape[1]
    char_emb_var = tf.Variable(emb_mat, trainable=True)
    with tf.variable_scope('Char_Emb_Network'):
        # Character embedding layer
        c_emb = tf.nn.embedding_lookup(char_emb_var, char_placeholder)

        # Character embedding network
        conv_results_list = []
        for filter_width in filter_widths:
            conv_results_list.append(tf.layers.conv2d(c_emb,
                                                      char_embedding_dim,
                                                      (1, filter_width),
                                                      padding='same',
                                                      kernel_initializer=INITIALIZER))
        units = tf.concat(conv_results_list, axis=3)
        units = tf.reduce_max(units, axis=2)
        if highway_on_top:
            sigmoid_gate = tf.layers.dense(units,
                                           1,
                                           activation=tf.sigmoid,
                                           kernel_initializer=INITIALIZER,
                                           kernel_regularizer=tf.nn.l2_loss)
            deeper_units = tf.layers.dense(units,
                                           tf.shape(units)[-1],
                                           kernel_initializer=INITIALIZER,
                                           kernel_regularizer=tf.nn.l2_loss)
            units = sigmoid_gate * units + (1 - sigmoid_gate) * deeper_units
            units = tf.nn.relu(units)
    return units
