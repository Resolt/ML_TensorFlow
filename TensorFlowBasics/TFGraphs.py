import tensorflow as tf

config = tf.ConfigProto(
	log_device_placement=True
)

with tf.Session(config=config) as sess:
	# DEFAULT GRAPH
	n1 = tf.constant(1)
	n2 = tf.constant(2)
	n3 = tf.add(n1, n2)
	print(sess.run(n3))

	# GET AND PRINT DEFAULT GRAPH
	dgraph = tf.get_default_graph()
	print(dgraph)

	# OWN GRAPH
	g = tf.Graph()

	# SET DEFAULT
	g.as_default()
	print(g is tf.get_default_graph())

		

print("test")


