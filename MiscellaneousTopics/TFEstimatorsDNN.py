from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split as TTS
from sklearn.preprocessing import MinMaxScaler as MMS
from sklearn.metrics import classification_report as CR
from sklearn.metrics import confusion_matrix as CM

import tensorflow as tf
from tensorflow import estimator

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

# FEATURE COLUMNS
feat_cols = [tf.feature_column.numeric_column('x',shape=[13])]

# MODEL
deep_model = estimator.DNNClassifier(
	hidden_units=[20, 20, 20, 20],
	feature_columns=feat_cols,
	n_classes=3,
	optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.001)
)

# INPUT FUNCTION
input_func = estimator.inputs.numpy_input_fn(x={'x': scaled_x_train,}, y=y_train, shuffle=True, batch_size=50, num_epochs=100)

# TRAINING
deep_model.train(input_fn=input_func, steps=500)

# EVALUATION
input_func_eval = estimator.inputs.numpy_input_fn(x={'x': scaled_x_test}, shuffle=False)

preds = list(deep_model.predict(input_fn=input_func_eval))

predictions = [p['class_ids'][0] for p in preds]

print(CR(y_test,predictions))
print(CM(y_test,predictions))