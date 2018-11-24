from sklearn.metrics import confusion_matrix as CM
from sklearn.metrics import classification_report as CR
from sklearn.preprocessing import MinMaxScaler as MMS
from sklearn.model_selection import train_test_split as TTS
from sklearn.datasets import load_wine

# import tensorflow as tf
from tensorflow.contrib.keras import models, layers, losses, metrics, activations, optimizers

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

# KERAS LAYERS
dnn_keras_model = models.Sequential()

dnn_keras_model.add(layers.Dense(units=20, input_dim=13, activation='relu'))
dnn_keras_model.add(layers.Dense(units=20, activation='elu'))
dnn_keras_model.add(layers.Dense(units=20, activation='elu'))
dnn_keras_model.add(layers.Dense(units=20, activation='elu'))
dnn_keras_model.add(layers.Dense(units=3, activation='softmax'))

# COMPILE MODEL
dnn_keras_model.compile(
	optimizer='Adadelta',
	loss='sparse_categorical_crossentropy',
	metrics=['accuracy']
)

# TRAIN
dnn_keras_model.fit(scaled_x_train,y_train,epochs=50)

# PREDICTIONS
predictions = dnn_keras_model.predict_classes(scaled_x_test)

# EVALUATION
print(CR(y_true=y_test,y_pred=predictions))