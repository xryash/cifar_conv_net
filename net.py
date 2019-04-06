import tensorflow as tf


def conv2d(x, W, b, strides=1):
    """Conv2d layer wrapper"""
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    """MaxPool layer wrapper"""
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')


def flatten(x):
    x_shape = x.get_shape().as_list()
    return tf.reshape(x, [-1, x_shape[1] * x_shape[2] * x_shape[3]])


def full(x, W, b, activation_fn=None, dropout_rate=0.2):
    """Fully connected layer wrapper"""
    x = tf.add(tf.matmul(x, W), b)
    if not activation_fn is None:
        x = activation_fn(x)
    if dropout_rate is 0.0:
        return x
    else:
        return tf.nn.dropout(x, rate=dropout_rate)


def net(x, biases, weights):
    conv_1 = conv2d(x, weights['w_conv_1'], biases['b_conv_1'])
    conv_1 = maxpool2d(conv_1, k=2)

    conv_2 = conv2d(conv_1, weights['w_conv_2'], biases['b_conv_2'])
    conv_2 = maxpool2d(conv_2, k=2)

    conv_3 = conv2d(conv_2, weights['w_conv_3'], biases['b_conv_3'])
    conv_3 = maxpool2d(conv_3, k=2)

    flatt = flatten(conv_3)

    fully_connected_1 = full(flatt, weights['w_full_1'], biases['b_full_1'], activation_fn=tf.nn.relu, dropout_rate=0.2)

    fully_connected_2 = full(fully_connected_1, weights['w_full_2'], biases['b_full_2'], activation_fn=tf.nn.relu,
                             dropout_rate=0.2)

    fully_connected_3 = full(fully_connected_2, weights['w_full_3'], biases['b_full_3'], activation_fn=tf.nn.relu,
                             dropout_rate=0.2)

    out = full(fully_connected_3, weights['out'], biases['out'], activation_fn=None, dropout_rate=0.0)

    return out
