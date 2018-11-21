import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data

# GET MNIST DATA
mnist = input_data.read_data_sets('/home/resolt/Workspaces/MachineLearning/MNIST_data',one_hot=True)

# HELPERS

# INIT WEIGHTS
def init_weights(shape):
	init_random_dist = tf.truncated_normal(shape,stddev=0.1)
	return tf.Variable(init_random_dist)

# INIT BIAS
def init_biases(shape):
	init_bias_vals = tf.constant(0.1,shape=shape)
	return tf.Variable(init_bias_vals)

# CONV2D
def conv2d(x,W):
	# x --> [batch, H, W, Channels]
	# W --> [filter_H, filter_W, Chanels_IN, Channels_OUT]
	return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')
	# STRIDES BY HOW MUCH WE SHIFT PER CONVOLUTION
	# PADDING SAME == PADDING WITH ZEROS

# POOLING
def max_pool_2by2(x):
	# x --> [batch,h,w,c]
	return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

# CONVOLUTIONAL LAYER
def conv_layer(input_x,shape):
	W = init_weights(shape)
	b = init_biases([shape[3]])
	return tf.nn.relu(tf.add(conv2d(input_x,W),b))

# DENSE LAYER
def dense_layer(input_layer,size):
	input_size = int(input_layer.get_shape()[1])
	W = init_weights([input_size,size])
	b = init_biases([size])
	return tf.add(tf.matmul(input_layer,W),b)

# PLACEHOLDERS
x = tf.placeholder(tf.float32,shape=[None,784]) # INPUT PLACEHOLDER
y_true = tf.placeholder(tf.float32,shape=[None,10]) # "y_train" PLACEHOLDER

# LAYERS
x_image = tf.reshape(x,[-1,28,28,1]) # RESHAPING THE INPUT "ARRAY" BACK INTO 2D SPACE

# COLVOLUTION LAYER 1 
convo_1 = conv_layer(x_image,shape=[5,5,1,32])
convo_1_pooling = max_pool_2by2(convo_1)

# CONVOLUTION LAYER 2
convo_2 = conv_layer(convo_1_pooling,shape=[5,5,32,64])
convo_2_pooling = max_pool_2by2(convo_2)
convo_2_flat = tf.reshape(convo_2_pooling,shape=[-1,7*7*64])

# DENSE LAYER
dense_1 = tf.nn.relu(dense_layer(convo_2_flat,1024))

# DROPOUT
hold_prob = tf.placeholder(tf.float32)
dense_1_drop = tf.nn.dropout(dense_1,keep_prob=hold_prob)

# OUT
y_pred = dense_layer(dense_1_drop,10)

# LOSS FUNCTION
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true,logits=y_pred))

# OPTIMIZER
optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
train = optimizer.minimize(cross_entropy)

# MATCHES
acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_pred,1),tf.argmax(y_true,1)),tf.float32))

# INITIALIZER
init = tf.global_variables_initializer()

steps = 2000

# SESSION
with tf.Session() as sess:
	sess.run(init)

	for i in range(steps):
		batch_x, batch_y = mnist.train.next_batch(200)
		sess.run(train,feed_dict={x:batch_x,y_true:batch_y,hold_prob:0.5})

		if i%100 == 0:
			accuracy = sess.run(acc,feed={x:mnist.test.images,y_true:mnist.test.labels,hold_prob:1.0})
			print("ON STEP: {}".format(i))
			print("ACCURACY: {}".format(accuracy))

	


