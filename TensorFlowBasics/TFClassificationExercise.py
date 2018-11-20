import numpy as np 
import pandas as pd 
import tensorflow as tf 
from sklearn.metrics import confusion_matrix as CM 
from sklearn.metrics import classification_report as CR
from sklearn.model_selection import train_test_split as TTS 

# READ AND ALTER THE DATA
dfile = '/home/resolt/Workspaces/MachineLearning/TensorFlow/TensorFlowBasics/census_data.csv'
df = pd.read_csv(dfile)

print(df.head())
print(df.info())

# OLD SCHOOLD FACTORIAZTION INTO NUMERIC VALUES
# for c in list(df.columns):
# 	if type(df[c][0]) == str:
# 		labels, uniques = pd.factorize(df[c])
# 		df[c] = labels

# MAKE INCOME BRACKET NUMBERS (0/1)
tcol = 'income_bracket'
labels, uniques = pd.factorize(df[tcol])
df[tcol] = labels

# TTS
X = df.drop(tcol,axis=1)
y = df[tcol]

X_train,X_test,y_train,y_test = TTS(X,y,test_size=0.3,random_state=64)

# FEATURE COLUMNS
fcols = [
	tf.feature_column.numeric_column(c) if type(df[c][0]) != str else
	tf.feature_column.embedding_column(tf.feature_column.categorical_column_with_hash_bucket(c,len(pd.factorize(X[c])[1])),dimension=len(pd.factorize(X[c])[1]))
	for c in list(X.columns)
	]

# INPUT FUNCTION
input_func_train = tf.estimator.inputs.pandas_input_fn(x=X_train,y=y_train,batch_size=300,num_epochs=100,shuffle=True)

# MODEL
model = tf.estimator.DNNClassifier(hidden_units=[50,100,100,50],feature_columns=fcols)

# TRAINING
model.train(input_fn=input_func_train,steps=None)

# EVALUATION
input_func_eval = tf.estimator.inputs.pandas_input_fn(x=X_test,shuffle=False,num_epochs=1)
preds = model.predict(input_fn=input_func_eval)
lpreds = list(preds)
cpreds = [pred['class_ids'][0] for pred in list(lpreds)]

print(CM(y_true=y_test,y_pred=cpreds))

print(CR(y_true=y_test,y_pred=cpreds))

