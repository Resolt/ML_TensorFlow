import tensorflow as tf

a = tf.add(1, 2,name="FirstAdd")

b = tf.add(3, 4, name="SecondAdd")

c = tf.multiply(a, b, name="Multiplication")

path = '/home/resolt/Workspaces/MachineLearning/BOARDS/test1'

with tf.Session() as sess:
	writer = tf.summary.FileWriter(path, sess.graph)
	print(sess.run(c))
	writer.close()
