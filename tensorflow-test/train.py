import tensorflow as tf
import numpy as np
IMG_SIZE_PX = 227
SLICE_COUNT = 20

n_classes = 5
batch_size = 10

x = tf.placeholder('float')
y = tf.placeholder('float')

keep_rate = 0.8

#定义一个函数，用于初始化所有的权值 W
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

#定义一个函数，用于初始化所有的偏置项 b
def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv3d(x, W):
    return tf.nn.conv3d(x, W, strides=[1, 1, 1, 1, 1], padding='SAME')

def maxpool3d(x):
    #                        size of window         movement of window as you slide about
    return tf.nn.max_pool3d(x, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='SAME')

def convolutional_neural_network(x):
    #                # 5 x 5 x 5 patches, 1 channel, 32 features to compute.
    weights = {'W_conv1': weight_variable([3, 3, 3, 1, 32]),
               #       5 x 5 x 5 patches, 32 channels, 64 features to compute.
               'W_conv2': weight_variable([3, 3, 3, 32, 64]),
               #                                  64 features
               'W_fc': weight_variable([57*57*5*64, 1024]),
               'out': weight_variable([1024, n_classes])}

    biases = {'b_conv1': bias_variable([32]),
               'b_conv2': bias_variable([64]),
               'b_fc': bias_variable([1024]),
               'out': bias_variable([n_classes])}

    #                            image X      image Y        image Z
    x = tf.reshape(x, shape=[-1, IMG_SIZE_PX, IMG_SIZE_PX, SLICE_COUNT, 1])

    print(x)
    h_conv1 = tf.nn.relu(conv3d(x, weights['W_conv1']) + biases['b_conv1'])
    print(h_conv1)
    h_pool1 = maxpool3d(h_conv1)
    print(h_pool1)


    h_conv2 = tf.nn.relu(conv3d(h_pool1, weights['W_conv2']) + biases['b_conv2'])
    print(h_conv2)
    h_pool2 = maxpool3d(h_conv2)
    print(h_pool2)

    h_pool2_flat = tf.reshape(h_pool2, [-1, 57*57*5*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, weights['W_fc'])+biases['b_fc'])
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_rate)

    prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, weights['out'])+biases['out'])

    return prediction


much_data = np.load('muchdata-227-227-20.npy')
# If you are working with the basic sample data, use maybe 2 instead of 100 here... you don't have enough data to really do this
train_data = much_data
validation_data = much_data


def train_neural_network(x):
    prediction = convolutional_neural_network(x)
    cross_entropy = -tf.reduce_sum(y * tf.log(prediction))
    # cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    # optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(cost)
    train_step = tf.train.GradientDescentOptimizer(1e-3).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    hm_epochs = 10
    with tf.Session() as sess:
        # sess.run(tf.global_variables_initializer())

        sess.run(tf.global_variables_initializer())

        for epoch in range(hm_epochs):
            for data in train_data:
                try:
                    X = data[0]
                    Y = data[1]
                    _, c = sess.run(train_step, feed_dict={x: X, y: Y})
                    print(c)
                except Exception as e:
                    # I am passing for the sake of notebook space, but we are getting 1 shaping issue from one
                    # input tensor. Not sure why, will have to look into it. Guessing it's
                    # one of the depths that doesn't come to 20.
                    pass
                    # print(str(e))


            # correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

            # print('Accuracy:', accuracy.eval({x: [i[0] for i in validation_data], y: [i[1] for i in validation_data]}))

        # print('Done. Finishing accuracy:')
        # print('Accuracy:', accuracy.eval({x: [i[0] for i in validation_data], y: [i[1] for i in validation_data]}))


        # Run this locally:
train_neural_network(x)