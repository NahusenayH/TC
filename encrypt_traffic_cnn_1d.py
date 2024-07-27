import tensorflow.compat.v1 as tf

import os
import sys
import input_data  # Make sure this module can handle your data format
import numpy as np

tf.disable_v2_behavior()
DATA_DIR = sys.argv[1]
TRAIN_ROUND = int(sys.argv[2])

# Simplified for two classes
CLASS_NUM = 10
LABELS = {0: 'Facebook', 1: 'icq', 2: 'linkedin', 3:'sftp', 4:'SkypeAudio', 5:'SkypeChat', 6:'SkypeFile', 7:'Spotify', 8:'Vimeo',9:'VOIP'}

sess = tf.InteractiveSession()

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('data_dir', DATA_DIR, 'Directory for storing data')

mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
print("Total training examples:", mnist.train._num_examples)
    
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 1, 3, 1], strides=[1, 1, 3, 1], padding='SAME')

def convert_labels_to_binary(labels):
    # Assuming labels with the first 5 classes (0-4) are for the first binary class
    # and the last 5 classes (5-9) are for the second binary class
    binary_labels = np.zeros((labels.shape[0], 2))
    binary_labels[:, 0] = np.sum(labels[:, :5], axis=1)
    binary_labels[:, 1] = np.sum(labels[:, 5:], axis=1)
    return binary_labels

x = tf.placeholder(tf.float32, shape=[None, 784])  # Assuming your images are 28x28 pixels
y_ = tf.placeholder(tf.float32, shape=[None, CLASS_NUM])

w_conv1 = weight_variable([1, 25, 1, 32])
b_conv1 = bias_variable([32])
x_image = tf.reshape(x, [-1, 1, 784, 1])
h_conv1 = tf.nn.relu(conv2d(x_image, w_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

w_conv2 = weight_variable([1, 25, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

w_fc1 = weight_variable([1*88*64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 1*88*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)

keep_prob = tf.placeholder("float")
rate = 1 - keep_prob 
h_fc1_drop = tf.nn.dropout(h_fc1, rate)

w_fc2 = weight_variable([1024, CLASS_NUM])
b_fc2 = bias_variable([CLASS_NUM])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, w_fc2) + b_fc2)
cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
train_step = tf.train.GradientDescentOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

sess.run(tf.global_variables_initializer())
for i in range(TRAIN_ROUND + 1):
    # print("num examples :",mnist.train.num_examples)
    print("train lenght : ", mnist.train)
    batch = mnist.train.next_batch(2)
    batch_labels = np.array(batch[1])
    binary_labels = np.zeros((batch_labels.shape[0], 2))    
    binary_labels[:, 0] = np.sum(batch_labels[:, :5], axis=1)  # Facebook
    binary_labels[:, 1] = np.sum(batch_labels[:, 5:], axis=1)  # LinkedIn

    print("Batch images shape:", np.array(batch[0]).shape)  # Should be (?, 784) for MNIST
    print("Batch labels shape:", np.array(batch[1]).shape)  # Should be (?, 2) for binary classification

    if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], rate: 1.0})
        print("step %d, training accuracy %g" % (i, train_accuracy))
    train_step.run(feed_dict={x: batch[0], y_: batch[1], rate: 0.5})

# Save the model after training
saver = tf.train.Saver()
model_path = "model/model.ckpt"
save_path = saver.save(sess, model_path)
print("Model saved in file: %s" % save_path)

# Load the model if needed
# saver.restore(sess, "model/model.ckpt")
# print("Model restored.")

# Evaluate the model on the test data
binary_test_labels = convert_labels_to_binary(mnist.test.labels)
test_accuracy = accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, rate: 1.0})
print("Test accuracy: %g" % test_accuracy)

# Generate a detailed classification report
predicted_labels = tf.argmax(y_conv, 1).eval(feed_dict={x: mnist.test.images, rate: 1.0})
actual_labels = tf.argmax(mnist.test.labels, 1).eval()
confusion_matrix = tf.math.confusion_matrix(actual_labels, predicted_labels)
print("Confusion Matrix:")
print(sess.run(confusion_matrix))

# Print more detailed statistics
precision = tf.metrics.precision(labels=actual_labels, predictions=predicted_labels)
recall = tf.metrics.recall(labels=actual_labels, predictions=predicted_labels)
sess.run(tf.local_variables_initializer())  # Initialize local variables for metrics calculation
precision_value, recall_value = sess.run([precision, recall])
print("Precision: %g, Recall: %g" % (precision_value[1], recall_value[1]))

sess.close()

