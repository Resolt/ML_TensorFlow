import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split as TTS
import tensorflow as tf

# READ THE DATA
diafile = 'TensorFlow/TensorFlowBasics/pima-indians-diabetes.csv'
dia = pd.read_csv(diafile)
print(dia.head())

# NORMALIZATION
cols_to_norm = list(dia.columns)[:-3]
dia[cols_to_norm] = dia[cols_to_norm].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
print(dia.head())

# GET THE FEATURE COLUMNS
fcnames = cols_to_norm
fcols = [tf.feature_column.numeric_column(c) for c in cols_to_norm]

# SOME SPECIAL EXAMPLES OF HOW HANDLE CATEGORIES AND HOW TO BUCKET NUMERIC DATA
assigned_group = tf.feature_column.categorical_column_with_vocabulary_list('Group',['A','B','C','D'])
# age_bucket = tf.feature_column.bucketized_column('Age',boundaries=[20,30,40,50,60,70,80])
age_column = tf.feature_column.numeric_column('Age')
age_bucket = tf.feature_column.bucketized_column(age_column,boundaries=[20,30,40,50,60,70,80])

# ADD THE SPECIAL CASES TO THE LIST
fcols.append(assigned_group)
fcols.append(age_bucket)

# TRAIN TEST SPLIT
X_data = dia.drop('Class',axis=1)
y_data = dia['Class']
X_train,X_test,y_train,y_test = TTS(X_data,y_data,test_size=0.3,random_state=101)

# INPUT FUNCTION
input_func = tf.estimator.inputs.pandas_input_fn(x=X_train,y=y_train,batch_size=10,num_epochs=1000,shuffle=True,num_threads=8)

# ESTIMATOR MODEL
model = tf.estimator.LinearClassifier(feature_columns=fcols,n_classes=2)

# TRAINING
model.train(input_fn=input_func,steps=1000)

# EVALUATION
eval_input_func = tf.estimator.inputs.pandas_input_fn(x=X_test,y=y_test,batch_size=10,num_epochs=1,shuffle=False)
results = model.evaluate(eval_input_func)
print(results)

# PREDICTIONS
pred_input_func = tf.estimator.inputs.pandas_input_fn(x=X_test,batch_size=10,num_epochs=1,num_threads=8,shuffle=False)
preds = model.predict(pred_input_func)
# lpreds = list(preds)
# print(lpreds)

# DNN CLASSIFIER

# DOESN'T WORK BECAUSE OF CATEGORICAL COLUMN
# dnn_model = tf.estimator.DNNClassifier(hidden_units=[10,10,10],feature_columns=fcols,n_classes=2)
# dnn_model.train(input_fn=input_func,steps=1000)

embedded_group_col = tf.feature_column.embedding_column(assigned_group,dimension=4)

fcols_dnn = [tf.feature_column.numeric_column(c) for c in cols_to_norm]
fcols_dnn.append(age_bucket)
fcols_dnn.append(embedded_group_col)

input_func_dnn = tf.estimator.inputs.pandas_input_fn(X_train,y_train,batch_size=10,num_epochs=1000,shuffle=True)
dnn_model = tf.estimator.DNNClassifier(hidden_units=[10,20,20,20,10],feature_columns=fcols_dnn,n_classes=2)
dnn_model.train(input_fn=input_func_dnn,steps=1000)

eval_input_func_dnn = tf.estimator.inputs.pandas_input_fn(x=X_test,y=y_test,batch_size=10,num_epochs=1,shuffle=False)

results_dnn = dnn_model.evaluate(eval_input_func_dnn)

print(results_dnn)

