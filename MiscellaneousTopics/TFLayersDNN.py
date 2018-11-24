from sklearn.metrics import confusion_matrix as CM
from sklearn.metrics import classification_report as CR
from sklearn.preprocessing import MinMaxScaler as MMS
from sklearn.model_selection import train_test_split as TTS
from sklearn.datasets import load_wine

import pandas as pd

# import tensorflow.layers as layers
import tensorflow as tf
from tensorflow.contrib.layers import fully_connected

# GET THE DATA
wine_data = load_wine()
feat_data = wine_data['data']
labels = wine_data['target']

# SPLIT
X_train, X_test, y_train, y_test = TTS(feat_data, labels, test_size=0.3, random_state=64)

# SCALE
scaler = MMS()
scaled_x_train = scaler.fit_transform(X_train)
scaled_x_test = scaler.transform(X_test)

# ONE HOT
onehot_y_train = pd.get_dummies(y_train).values
onehot_y_test = pd.get_dummies(y_test).values

# CONSTANTS
num_feat = X_train.shape[1]
num_neurons_in_hidden = 20
num_outputs = onehot_y_train.shape[1]
learning_rate = 0.001

# PLACEHOLDERS
X = tf.placeholder(tf.float32,shape=[None,num_feat])
y_true = tf.placeholder(tf.float32,shape=[None,num_outputs])

# ACTIVATION
actf = tf.nn.relu

# LAYERS
hidden1 = fully_connected(X, num_neurons_in_hidden, activation_fn=actf)
hidden2 = fully_connected(hidden1, num_neurons_in_hidden, activation_fn=actf)
hidden3 = fully_connected(hidden2, num_neurons_in_hidden, activation_fn=actf)
output = fully_connected(hidden3, num_outputs)

# layers = fully_connected(X, num_neurons_in_hidden, activation_fn=actf)
# layers = fully_connected(layers, num_neurons_in_hidden, activation_fn=actf)
# layers = fully_connected(layers, num_neurons_in_hidden, activation_fn=actf)
# layers = fully_connected(layers, num_outputs)

# LOSS
loss = tf.losses.softmax_cross_entropy(onehot_labels=y_true, logits=output)

# OPTIMIZER
optimizer = tf.train.AdamOptimizer(learning_rate)
train = optimizer.minimize(loss)

# INIT
init = tf.global_variables_initializer()

training_steps = 1000

# SESSION
with tf.Session() as sess:
	sess.run(init)

	for i in range(1, training_steps + 1):
		sess.run(train, feed_dict={X: scaled_x_train, y_true: onehot_y_train})


	# EVALUATION
	logits = output.eval(feed_dict={X: scaled_x_test})
	preds = tf.argmax(logits, axis=1)
	predictions = preds.eval()


print(CR(y_pred=predictions,y_true=y_test))
print(CM(y_pred=predictions,y_true=y_test))