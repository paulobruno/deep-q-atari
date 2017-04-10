import tensorflow as tf


# NN architecture
conv_width = 5
conv_height = 5
features_layer1 = 8
features_layer2 = 16
fc_num_outputs = 256

# Q-learning settings
learning_rate = 0.00025
dropout_keep_prob = 0.8


# ceil of a division, source: http://stackoverflow.com/questions/14822184/is-there-a-ceiling-equivalent-of-operator-in-python
def ceildiv(a, b):
    return -(-a // b)


def create_network(session, num_available_actions, game_resolution, img_channels):
    """ creates the network with 
    conv_relu + max_pool + conv_relu + max_pool + fc + dropout + fc """

    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


    s1_ = tf.placeholder(tf.float32, [None] + list(game_resolution) + [img_channels], name='State')

    target_q_ = tf.placeholder(tf.float32, [None, num_available_actions], name='TargetQ')

    # first convolutional layer
    W_conv1 = weight_variable([conv_height, conv_width, img_channels, features_layer1])
    b_conv1 = bias_variable([features_layer1])

    h_conv1 = tf.nn.relu(conv2d(s1_, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    # second convolutional layer
    W_conv2 = weight_variable([conv_height, conv_width, features_layer1, features_layer2])
    b_conv2 = bias_variable([features_layer2]) 

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    # densely connected layer
    W_fc1 = weight_variable([ceildiv(game_resolution[0],4)*ceildiv(game_resolution[1],4)*features_layer2, fc_num_outputs])
    b_fc1 = bias_variable([fc_num_outputs])

    h_pool2_flat = tf.reshape(h_pool2, [-1, ceildiv(game_resolution[0],4)*ceildiv(game_resolution[1],4)*features_layer2])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # dropout
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # readout layer
    W_fc2 = weight_variable([fc_num_outputs, num_available_actions])
    b_fc2 = bias_variable([num_available_actions])

    q = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    best_a = tf.argmax(q, 1)

    loss = tf.losses.mean_squared_error(q, target_q_)

    optimizer = tf.train.RMSPropOptimizer(learning_rate)

    train_step = optimizer.minimize(loss)

    def function_learn(s1, target_q):
        feed_dict = {s1_: s1, target_q_: target_q, keep_prob: dropout_keep_prob}
        l, _ = session.run([loss, train_step], feed_dict=feed_dict)
        return l

    def function_get_q_values(state):
        return session.run(q, feed_dict={s1_: state, keep_prob: dropout_keep_prob})

    def function_get_best_action(state):
        return session.run(best_a, feed_dict={s1_: state, keep_prob: dropout_keep_prob})

    def function_simple_get_best_action(state):
        return function_get_best_action(state.reshape([1, game_resolution[0], game_resolution[1], 1]))[0]
    
    return function_learn, function_get_q_values, function_simple_get_best_action
