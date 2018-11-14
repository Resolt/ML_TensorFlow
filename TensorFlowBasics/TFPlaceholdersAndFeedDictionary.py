import numpy as np
import tensorflow as tf

config = tf.ConfigProto(
	log_device_placement=True
)

np.random.seed(101)
tf.set_random_seed(101)

rand_a = np.random.uniform(0,100,(5,5))
rand_b = np.random.uniform(0,100,(5,1))

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)

add_op = tf.add(a,b)
mul_op = tf.multiply(a,b)

with tf.Session(config=config) as sess:
	add_result = sess.run(add_op,feed_dict={a:rand_a,b:rand_b})
	mul_result = sess.run(mul_op,feed_dict={a:rand_a,b:rand_b})
	

print(add_result)
print(mul_result)



