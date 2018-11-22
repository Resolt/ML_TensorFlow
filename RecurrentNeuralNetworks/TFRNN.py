import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

class TimeSeriesData():
	
	def __init__(self, num_points, xmin, xmax):
		self.xmin = xmin
		self.xmax = xmax
		self.num_points = num_points
		self.resolution = (self.xmax - self.xmin) / self.num_points
		self.x_data = np.linspace(xmin, xmax, num_points)
		self.y_true = np.sin(self.x_data)

	def ret_true(sefl, x_series):
		return np.sin(x_series)

	def next_batch(self, batch_size, steps, return_batch_ts=False):
		# GRAB RANDOM STARTING POINT FOR EACH BATCH OF DATA
		rand_start = np.random.rand(batch_size, 1)
		
		# CONVERT TO BE ON TIME SERIES
		ts_start = rand_start * (self.xmax - self.xmin - (steps * self.resolution))

		# CREATE BATCH TIME SERIES ON X AXIS
		batch_ts = ts_start + np.arange(0.0,steps+1) * self.resolution

		# CREATE THE Y DATA FOR THE TIME SERIES X AXIS FROM PREIVOUS STEP
		y_batch = np.sin(batch_ts)

		# FORMATTING FOR RNN
		if return_batch_ts:
			return y_batch[:, :-1].reshape(-1, steps, 1), y_batch[:, 1:].reshape(-1, steps, 1), batch_ts
		else:
			return y_batch[:, :-1].reshape(-1, steps, 1), y_batch[:, 1:].reshape(-1, steps, 1)


# SETTING
plotting = False

# CONSTANTS
num_inputs = 1
num_neurons = 200
num_outputs = 1
learning_rate = 0.0001
num_train_iterations = 5000
batch_size = 20
num_time_steps = 30

# INSTANTIATE DATA OBJECT
ts_data = TimeSeriesData(250, 0, 10)

# EXAMPLE OF WHAT WE GET FROM THE DATA OBJECT
y1, y2, ts = ts_data.next_batch(1, num_time_steps, True)

if plotting:
	plt.plot(ts_data.x_data, ts_data.y_true, label='Sin(t)')
	plt.plot(ts.flatten()[1:], y2.flatten(), 'o', label="Single Training Instance")
	plt.legend()
	plt.show()

# TRAINING DATA
train_inst = np.linspace(5, 5 + ts_data.resolution * (num_time_steps + 1), num_time_steps + 1)

if plotting:
	plt.plot(train_inst[:-1], ts_data.ret_true(train_inst[:-1]), 'bo', markersize=15, alpha=0.5, label="INSTANCE")
	plt.plot(train_inst[1:],ts_data.ret_true(train_inst[1:]),'ko',markersize=7,label='TARGET')
	plt.title("A TRAINING INSTANCE")
	plt.show()

# CREATING THE MODEL
tf.reset_default_graph()

# PLACEHOLDERS
X = tf.placeholder(tf.float32, [None, num_time_steps, num_inputs])
y = tf.placeholder(tf.float32, [None, num_time_steps, num_outputs])

# RNN CELL LAYER
cell = tf.contrib.rnn.BasicRNNCell(num_units=num_neurons, activation=tf.nn.relu)
# cell = tf.contrib.rnn.GRUCell(num_units=num_neurons, activation=tf.nn.relu)
# cell = tf.keras.layers.SimpleRNNCell(units=num_neurons, activation=tf.nn.relu)
cellw = tf.contrib.rnn.OutputProjectionWrapper(cell, output_size=num_outputs)

outputs, states = tf.nn.dynamic_rnn(cellw, X, dtype=tf.float32)

# MSE
loss_func = tf.reduce_mean(tf.square(tf.subtract(outputs, y)))

# OPTIMIZER
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train = optimizer.minimize(loss_func)

# INITIALIZER
init = tf.global_variables_initializer()

# SAVER
saver = tf.train.Saver()
savedir = '/home/resolt/Workspaces/MachineLearning/MODELS/rnn_timeseires_codealong/'

# SESSION TRAINING
with tf.Session() as sess:
	sess.run(init)

	for iteration in range(1, num_train_iterations+1):
		X_batch, y_batch = ts_data.next_batch(batch_size, num_time_steps)
		sess.run(train, feed_dict={X: X_batch, y: y_batch})
		
		if iteration % 100 == 0:
			mse = loss_func.eval(feed_dict={X: X_batch, y: y_batch})
			# print("ITERATION: {}\tMSE: {}".format(iteration, str(round(mse,8))[:8]))
			print("ITERATION: {}\tMSE: {:8f}".format(iteration, mse))
			
	
	saver.save(sess,savedir)

# SESSION PREDICTION
with tf.Session() as sess:
	
	saver.restore(sess, savedir)
	
	X_new = np.sin(np.array(train_inst[:-1].reshape(-1, num_time_steps, num_inputs)))
	y_pred = sess.run(outputs, feed_dict={X: X_new})
	

# PLOTTING
# TRAINING INSTANCE
plt.plot(train_inst[:-1], np.sin(train_inst[:-1]), 'bo', markersize=15, alpha=0.5, label="TRAINING INSTANCE")
# TARGET
plt.plot(train_inst[1:], np.sin(train_inst[1:]), 'ko', markersize=10, label='TARGET')
# PREDICTIONS
plt.plot(train_inst[1:],y_pred[0,:,0],'r.',markersize=10,label="PREDICTIONS")

plt.xlabel("TIME")
plt.legend()
plt.tight_layout()
plt.title("TESTING THE MODEL")
plt.show()

# SEED
training_instance = list(ts_data.y_true[:num_time_steps])

# NEW SERIES
with tf.Session() as sess:
	
	# RESTORE MODEL
	saver.restore(sess, savedir)

	for iteration in range(len(ts_data.x_data) - num_time_steps):
		X_batch = np.array(training_instance[-num_time_steps:]).reshape(1, num_time_steps, 1)
		y_pred = sess.run(outputs, feed_dict={X: X_batch})
		
		training_instance.append(y_pred[0, -1, 0])
		

plt.plot(ts_data.x_data, training_instance, 'b-')
plt.plot(ts_data.x_data[:num_time_steps], training_instance[:num_time_steps], 'r-', linewidth=3)
plt.xlabel("TIME")
plt.ylabel("Y")
plt.title("NEWSERIES")
plt.tight_layout()
plt.show()
