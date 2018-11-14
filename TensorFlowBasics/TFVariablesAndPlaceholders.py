import tensorflow as tf

config = tf.ConfigProto(
	log_device_placement=True
)

with tf.Session(config=config) as sess:
	# CREATE TENSOR (THE DATA)
	tensor = tf.random.uniform((100,100),0,1) # FLOAT32
	# CREATE VARIABLES
	variable1 = tf.Variable(initial_value=tensor)
	variable2 = tf.Variable(initial_value=tensor)
	# PLACEHOLDER
	placeholder = tf.placeholder(tf.float32,shape=(100,100))
	# OPERATION
	op = variable1 * variable2
	# VARIABLE INITIALIZER
	init = tf.global_variables_initializer()
	# INITIALIZE VARIABLES IN SESSION
	sess.run(init)
	# RUN VARIABLE
	print(sess.run(op))




