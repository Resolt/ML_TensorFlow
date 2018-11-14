import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from sklearn.model_selection import train_test_split as TTS

# ONE MILLION POINTS
data_size = 1000000
x_data = np.linspace(0.0,10.0,data_size).astype('float32')

# NOISE
noise = np.random.randn(len(x_data))

# THE SLOPE AND INTERCEPT WE'RE TRYING TO ARRIVE AT
slope = 0.5
intercept = 5

# THE TARGET DATA (WITH NOISE)
y_true = (slope * x_data) + 5 + noise

# CREATE DATAFRAME
x_df = pd.DataFrame(data=x_data,columns=['X'])
y_df = pd.DataFrame(data=y_true,columns=['Y'])

my_data = pd.concat([x_df,y_df],axis=1)

# my_data.sample(n=250).plot(kind='scatter',x='X',y='Y')
# plt.show()

# TF
batch_size = 100
batches = 10000
lrate = 0.00001

# CONFIG
config = tf.ConfigProto(
	log_device_placement=True
)

# VARIABLES
r = np.random.randn(2).astype('float32')
m = tf.Variable(r[0])
b = tf.Variable(r[1])

# PLACEHOLDERS FOR INPUT AND OUTPUT
xph = tf.placeholder(tf.float32,[batch_size])
yph = tf.placeholder(tf.float32,[batch_size])

# GRAPH
y_model = tf.add(tf.multiply(m, xph), b)

# LOSS FUNCTION
error = tf.reduce_sum(tf.square(tf.subtract(yph,y_model)))

# OPTIMIZER
optimizer = tf.train.GradientDescentOptimizer(learning_rate = lrate)
train = optimizer.minimize(error)

# VARIABLE INITIALIZER
init = tf.global_variables_initializer()

# RANDOM INDEX OPERATOR
ranop = tf.random_uniform(dtype=tf.int32, minval=0, maxval=data_size, shape=[batch_size])

with tf.Session(config=config) as sess:
	# INITALIZER VARIABLES
	sess.run(init)

	for i in range(batches):
		rand_ind = sess.run(ranop)

		feed = {xph:x_data[rand_ind],yph:y_true[rand_ind]}

		sess.run(train,feed_dict=feed)
	
	model_m,model_b = sess.run([m,b])

	y_hat = sess.run(tf.add(tf.multiply(x_data,model_m),model_b))


print(model_m,model_b)

my_data.sample(250).plot(kind='scatter',x='X',y='Y')
plt.plot(x_data,y_hat,'r')
plt.show()

