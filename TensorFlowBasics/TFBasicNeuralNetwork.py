import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split as TTS

import tensorflow as tf

plt.style.use('ggplot')

# CONFIG
config = tf.ConfigProto(
	log_device_placement=True
)

# SETTINGS
data_size = 100
show_size = 100
training_steps = 10000
lrate = 0.00005

# DATA - LINEAR WITH RANDOM NOISE
noise = data_size*0.05
x_data = np.linspace(0,data_size,data_size) + np.random.uniform(-noise,noise,data_size)
y_data = np.linspace(0,data_size,data_size) + np.random.uniform(-noise,noise,data_size)

# RANDOM STARTING VALUES FOR WEIGHT AND BIAS
r = np.random.uniform(-1,1,2)
m = tf.Variable(r[0])
b = tf.Variable(r[1])

# LOSS FUNCTION
# CREATE ERROR VARIABLE - THIS IS ESSENTIALLY AN OPERATOR WHICH CALCULATES THE ERROR FOR WHICH TO CORRECT
# error = 0
# for y,x in zip(x_data,y_data):
# 	y_hat = m*x + b
# 	error += (y-y_hat)**2

# error = tf.nn.l2_loss(tf.subtract(y_data, tf.add(tf.multiply(m,x_data), b))) # DON'T REALLY KNOW WHAT LOSS FUNCTION THIS IS
# error = tf.squared_difference(y_data, tf.add(tf.multiply(m,x_data), b)) # MEAN SQUARED ERROR
error = tf.reduce_mean(tf.squared_difference(y_data, tf.add(tf.multiply(m,x_data), b))) # REDUCED MEAN SQUARED ERROR

# INSTATIATE TENSORFLOWS GRADIENT DESCENT ALGORITHM WITH A SET LEARNING RATE
optimizer = tf.train.GradientDescentOptimizer(learning_rate=lrate)
# GET TRAINING OPERATOR AS THE GRADIENT DESCENT MINIMIZER TARGETING THE ERROR OPERATOR
train = optimizer.minimize(error)

# INSTANTIATE INITIALIZER
init = tf.global_variables_initializer()

with tf.Session(config=config) as sess:
	# INITIALIZE VARIABLES
	sess.run(init)

	# HOW MANY TIMES TO ITERATE OVER THE TRAINING DATA

	# ITERATE OVER THE TRAINING DATA
	for i in range(training_steps):
		sess.run(train)

	# GET THE FINAL SLOPE AND INTERCEPT (THAT IS WHAT WE'RE AFTER AFTER ALL)
	final_slope, final_intercept = sess.run([m,b])

# PRINT
print(final_slope, final_intercept)

# PLOT THE RESULTS
x_test = np.linspace(0,show_size,show_size)
y_pred = final_slope * x_test + final_intercept

plt.plot(x_test,y_pred,'r')
plt.plot(x_data[0:show_size],y_data[0:show_size],'o')
plt.show()



