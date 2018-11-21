import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# READ THE DATA
CIFAR_DIR = '/home/resolt/Workspaces/MachineLearning/CIFAR_data/'

def unpickle(file):
	import pickle
	with open(file, 'rb') as fo:
		cifar_dict = pickle.load(fo, encoding='bytes')
	
	return cifar_dict


dirs = ['batches.meta', 'data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5', 'test_batch']
all_data = [unpickle(CIFAR_DIR + direc) for direc in dirs]

batch_meta = all_data[0]
data_batch1 = all_data[1]
data_batch2 = all_data[2]
data_batch3 = all_data[3]
data_batch4 = all_data[4]
data_batch5 = all_data[5]
test_batch = all_data[6]

# PLOT ONE OF THE IMAGES

# RESHAPING INTO IMAGES
# X = data_batch1[b'data']
# X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype('uint8')

# PLOT
# plt.imshow(X[0])
# plt.show()

# CIFAR HELPERS

def one_hot_encode(vec, vals=10):  # ENCODING OF THE 10 POSSIBLE LABELS
	n = len(vec)
	out = np.zeros((n, vals))
	out[range(n), vec] = 1
	return out

class CifarHelper():

	def __init__(self):
		self.i = 0

		# GRAB A LIST OF ALL THE BATCHES FOR TRAINING
		self.all_train_batches = [data_batch1, data_batch2, data_batch3, data_batch4, data_batch5]
		# GRAB A LIST OF ALL THE BATHCES FOR TESTING
		self.test_batch = [test_batch]

		# INITALIZER SOME EMPTY VARIABLES FOR LATER
		self.training_images = None
		self.training_labels = None

		self.test_images = None
		self.test_labels = None

	def set_up_images(self):
		print("SETTING UP TRAINING DATA")

		# VERTICALLY STACK IMAGES
		self.training_images = np.vstack([d[b'data'] for d in self.all_train_batches])
		train_len = len(self.training_images)

		# RESHAPE AND NORMALIZE TRAINING IMAGES
		self.training_images = self.training_images.reshape(train_len, 3, 32, 32).transpose(0, 2, 3, 1)/255
		# ONE HOT ENCODE TRAINING LABELS
		self.training_labels = one_hot_encode(np.hstack([d[b'labels'] for d in self.all_train_batches]), 10)

		print("SETTING UP TEST DATA")

		# VERTICALLY STACK IMAGES
		self.test_images = np.vstack([d[b'data'] for d in self.test_batch])
		test_len = len(self.test_images)

		# RESHAPE AND NORMALIZE TEST IMAGES
		self.test_images = self.test_images.reshape(test_len, 3, 32, 32).transpose(0, 2, 3, 1) / 255
		# ONE HOT ENCODE TEST LABELS
		self.test_labels = one_hot_encode(np.hstack([d[b'labels'] for d in self.test_batch]), 10)

	def next_batch(self, batch_size):
		x = self.training_images[self.i : self.i + batch_size].reshape(batch_size, 32, 32, 3)
		y = self.training_labels[self.i : self.i + batch_size]
		self.i = (self.i + batch_size) % len(self.training_images)  # THIS INCREMENTATION ALLOWS LOOPING BACK AROUND TO THE START
		return x, y

# MODEL HELPERS

# INIT WEIGHTS
def init_weights(shape):
	init_random_dist = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(init_random_dist)

# INIT BIAS
def init_biases(shape):
	init_bias_vals = tf.constant(0.1, shape=shape)
	return tf.Variable(init_bias_vals)

# CONV2D
def conv2d(x, W):
	# x --> [batch, H, W, Channels]
	# W --> [filter_H, filter_W, Chanels_IN, Channels_OUT]
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
	# STRIDES BY HOW MUCH WE SHIFT PER CONVOLUTION
	# PADDING SAME == PADDING WITH ZEROS

# POOLING
def max_pool_2by2(x):
	# x --> [batch,h,w,c]
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# CONVOLUTIONAL LAYER
def conv_layer(input_x, shape):
	W = init_weights(shape)
	b = init_biases([shape[3]])
	return tf.nn.relu(tf.add(conv2d(input_x, W), b))

# DENSE LAYER
def dense_layer(input_layer, size):
	input_size = int(input_layer.get_shape()[1])
	W = init_weights([input_size, size])
	b = init_biases([size])
	return tf.add(tf.matmul(input_layer, W), b)



# INSTANTIATE THE HELPER CLASS AND RUN THE SETUP
CH = CifarHelper()
CH.set_up_images()

# PLACEHOLDERS
x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
y_true = tf.placeholder(tf.float32, shape=[None, 10])
hold_prob = tf.placeholder(tf.float32)

# CREATE THE LAYERS
conv_1 = conv_layer(x, shape=[4, 4, 3, 32]) # CONVOLUTE IN 4 BY 4 SECTIONS
conv_1_pool = max_pool_2by2(conv_1) # SOFT MAX POOL

conv_2 = conv_layer(conv_1_pool, shape=[3, 3, 32, 64]) # CONVOLUTE IN 3 BY 3 SECTIONS
conv_2_pool = max_pool_2by2(conv_2)

resh = tf.reshape(conv_2_pool, shape=[-1, 8 * 8 * 64])  # MAKE THE TENSOR ONEDIMENSIONAL AGAIN

dense_1 = tf.nn.relu(dense_layer(resh, 1024))
dense_1_drop = tf.nn.dropout(dense_1, keep_prob=hold_prob)

dense_2 = tf.nn.relu(dense_layer(dense_1_drop, 512))
dense_2_drop = tf.nn.dropout(dense_2, keep_prob=hold_prob)

y_pred = dense_layer(dense_2_drop, 10)

# LOSS FUNCTION
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred))

# SETTINGS
steps = 10000
batch_size = 1000
hprob = 0.25
lrate = 0.0003

# OPTIMIZER
optimizer = tf.train.AdamOptimizer(learning_rate=lrate)
train = optimizer.minimize(cross_entropy)

# ACCURACY
acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1)), tf.float32))

# VARIABEL INITIALIZER
init = tf.global_variables_initializer()

# SESSION
with tf.Session() as sess:
	sess.run(init)

	for i in range(1, steps + 1):
		batch_x, batch_y = CH.next_batch(batch_size)
		sess.run(train, feed_dict={x: batch_x, y_true: batch_y, hold_prob: hprob})
		
		if i % 100 == 0:
			accuracy = sess.run(acc, feed_dict={x: CH.test_images, y_true: CH.test_labels, hold_prob: 1.0})
			print("STEP {}\tACCURACY {}".format(i, accuracy))


