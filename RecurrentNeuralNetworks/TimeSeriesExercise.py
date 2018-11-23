import numpy as np
import pandas as pd 
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler as SS

# READ DATA
dfile = '/home/resolt/Workspaces/MachineLearning/TensorFlow/RecurrentNeuralNetworks/monthly-milk-production.csv'
milk = pd.read_csv(dfile, index_col='Month')

# MAKE INDEX TIME SERIES
milk.index = pd.to_datetime(milk.index)

# PLOT OUT THE TIMESERIES DATA
# plt.plot(milk.index,milk['Milk Production'])
# plt.show()

# TRAIN TEST SPLIT
# X = milk['Milk Production'].head(milk.shape[1]-13)
# y = milk['Milk Production'].tail(12)
X_data = milk.head(milk.shape[0]-13)
y_data = milk.tail(12)

# SCALING
scaler = SS()
scaler.fit(X_data)
X_data = scaler.transform(X_data)
y_data = scaler.transform(y_data)

# FEEDER
def next_batch(training_data, batch_size, steps):
	"""
	INPUT: Data, Batch Size, Time Steps per batch
	OUTPUT: A tuple of y time series results. y[:,:-1] and y[:,1:]
	"""

	# STEP 1: Use np.random.randint to set a random starting point index for the batch.
	# Remember that each batch needs have the same number of steps in it.
	# This means you should limit the starting point to len(data)-steps
	rand_start = np.random.randint(low=0, high=len(training_data) - steps)

	# STEP 2: Now that you have a starting index you'll need to index the data from
	# the random start to random start + steps + 1. Then reshape this data to be (1,steps+1)
	rand_data = np.array(training_data[rand_start:rand_start+steps+1]).reshape(1,steps+1)

	# STEP 3: Return the batches. You'll have two batches to return y[:,:-1] and y[:,1:]
	# You'll need to reshape these into tensors for the RNN to .reshape(-1,steps,1)
	a = rand_data[:, :-1].reshape(-1, steps, 1)
	b = rand_data[:, 1:].reshape(-1, steps, 1)
	
	return a, b

# SETTINGS
num_inputs = 1
num_time_steps = 12
num_neurons = 100
num_outputs = 1
learning_rate = 0.03
train_iters = 4000
batch_size = 1

# PLACEHOLDERS
X = tf.placeholder(tf.float32, shape=[None, num_time_steps, num_inputs])
y = tf.placeholder(tf.float32, shape=[None, num_time_steps, num_outputs])

# RNN LAYER
cell = tf.contrib.rnn.BasicRNNCell(num_units=num_neurons, activation=tf.nn.relu)
cellw = tf.contrib.rnn.OutputProjectionWrapper(cell, output_size=num_outputs)

outputs, states = tf.nn.dynamic_rnn(cellw, X, dtype=tf.float32)

# MSE
loss_func = tf.reduce_mean(tf.square(tf.subtract(outputs, y)))

# OPTIMIZER
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train = optimizer.minimize(loss_func)

# INITALIZER
init = tf.global_variables_initializer()

# SAVER
saver = tf.train.Saver()
savedir = savedir = '/home/resolt/Workspaces/MachineLearning/MODELS/rnn_timeseries_excercise/'

# SESSION
with tf.Session() as sess:
	sess.run(init)

	for iteration in range(1, train_iters+1):
		X_batch, y_batch = next_batch(training_data = X_data, batch_size = 1, steps = num_time_steps)
		sess.run(train, feed_dict={X: X_batch, y: y_batch})

		if iteration % 100 == 0:
			mse = loss_func.eval(feed_dict={X: X_batch, y: y_batch})
			# print("ITERATION: {}\tMSE: {}".format(iteration, str(round(mse,8))[:8]))
			print("ITERATION: {}\tMSE: {:8f}".format(iteration, mse))

	saver.save(sess,savedir)

# PREDICTION
print(y_data)

train_inst = np.array(X_data[-num_time_steps:]).reshape(-1, num_time_steps, 1)

with tf.Session() as sess:

	saver.restore(sess, savedir)
	
	for i in range(12):
		print(i)
		y_pred = sess.run(outputs, feed_dict={X: train_inst[:,-num_time_steps:,:]})
		train_inst = np.append(train_inst, y_pred[0,-1:,0]).reshape(-1,num_time_steps+i+1,num_inputs)


print(train_inst[0, -num_time_steps:, :])

# THIS SUCKS - SKIPPING AHEAD