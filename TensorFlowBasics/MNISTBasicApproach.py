import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# READ THE MNIST DATA
mnist = input_data.read_data_sets('/home/resolt/Workspaces/MachineLearning/MNIST_data',one_hot=True)

print(mnist.train.num_examples)
print(mnist.test.num_examples)

# SHOW AN IMAGE IN PLT
# single_image = mnist.train.images[1].reshape(28,28)
# plt.imshow(single_image,cmap='gist_gray')
# plt.show()

# ALREADY NORMALIZED
# print(single_image.min())
# print(single_image.max())

# PLACHOLDERS
x = tf.placeholder(tf.float32,shape=[None,784])

# VARIABLES
W = tf.Variable(tf.zeros([784,10])) # WEIGHTS
b = tf.Variable(tf.zeros([10])) # BIASES

# CREATE GRAPH OPERTAIONS
y = tf.add(tf.matmul(x,W), b)

# LOSS FUNCTION
y_true = tf.placeholder(tf.float32,[None,10]) # FOR THE "y_train"
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true,logits=y))

# OPTIMIZER
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5)
train = optimizer.minimize(cross_entropy) # FROM THE GRADIENT DESCENT INSTANTINIATION WE INSTANTIATE A MINIMIZER - THIS AUTOMATICALLY FINDS AND ADJUST THE tf.Variable TENSORS AND ADJUSTS THEM IN ORDER TO MINIMIZE THE OUTPUT OF OUR DEFINED LOSS FUNCTION (MEANED SOFT-MAX IMPORTED FROM TENSORFLOW)

# INITALIZER
init = tf.global_variables_initializer()

# TENSOR FOR CORRECT PREDICTIONS - ESSENTIALLY IT'S OWN GRAPH FOR CALCULATING ACCURACY
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_true,1))
acc = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

# CREATE SESSION
with tf.Session() as sess:
	# INITIALIZE THE GRAPH
	sess.run(init)

	# STEP THROUGH BATCHES AND TRAIN
	for step in range(1000):
		batch_x, batch_y = mnist.train.next_batch(500)
		sess.run(train,feed_dict={x:batch_x,y_true:batch_y})

	# EVALUATE
	accuracy = sess.run(acc,feed_dict={x:mnist.test.images,y_true:mnist.test.labels})
	

print(accuracy)