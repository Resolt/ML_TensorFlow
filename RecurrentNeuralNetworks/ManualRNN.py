import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# CONSTANTS
num_inputs = 2
num_neurons = 3

# PLACEHOLDERS
x0 = tf.placeholder(tf.float32, shape=[None, num_inputs])
x1 = tf.placeholder(tf.float32, shape=[None, num_inputs])

# VARIABLES
Wx = tf.Variable(tf.random_normal(shape=[num_inputs, num_neurons]))
Wy = tf.Variable(tf.random_normal(shape=[num_neurons, num_neurons]))
b = tf.Variable(tf.zeros([1, num_neurons]))

# GRAPHS
y0 = tf.tanh(tf.add(tf.matmul(x0, Wx), b))
y1 = tf.tanh(tf.add(tf.matmul(y0, Wy), tf.add(tf.matmul(x1, Wx), b)))

# INITIALIZER
init = tf.global_variables_initializer()

# DATA
x0_batch = np.array([[0, 1], [2, 3], [4, 5]])
x1_batch = np.array([[100, 101], [102, 103], [104, 105]])

with tf.Session() as sess:
	sess.run(init)

	y0_output_vals, y1_output_vals = sess.run([y0, y1], feed_dict={x0: x0_batch, x1: x1_batch})
	
print(y0_output_vals)
print(y1_output_vals)
