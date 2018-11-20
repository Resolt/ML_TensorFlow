import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.model_selection import train_test_split as TTS 
import tensorflow as tf 
from sklearn.preprocessing import minmax_scale as MMS

# READ DATA
dfile = 'TensorFlow/TensorFlowBasics/cal_housing_clean.csv'
df = pd.read_csv(dfile)

# SCALE AND TRAIN TEST SPLIT
tcol = 'medianHouseValue'
scaled = MMS(df.drop(tcol,axis=1),feature_range=(0,1),copy=True)
X = pd.DataFrame(scaled,columns=list(df.drop(tcol,axis=1).columns))
y = df[tcol]
X_train,X_test,y_train,y_test = TTS(X,y,test_size=0.3,random_state=64)

# CREATE FEATURE COLUMNS
fcols = [tf.feature_column.numeric_column(c) for c in list(X.columns)]
print(fcols)

# INPUT FUNCTION
input_func_train = tf.estimator.inputs.pandas_input_fn(x=X_train,y=y_train,num_epochs=10000,num_threads=8,batch_size=1000,shuffle=True)

# DNN REGERSSOR
model = tf.estimator.DNNRegressor(hidden_units=[4608,900,90,9,9,9,9,9,9],feature_columns=fcols)

# TRAIN
model.train(input_fn=input_func_train,steps=10000)

# PREDICT
input_func_pred = tf.estimator.inputs.pandas_input_fn(x=X_test,num_epochs=1,batch_size=10,shuffle=False)
preds = model.predict(input_fn=input_func_pred)
lpreds = [a['predictions'][0] for a in list(preds)]

# RMSE
err = y_test-lpreds # ERROR - E
s = err**2 # SQUARED        - S
m = np.mean(s) # MEAN       - M
r = np.sqrt(m) # ROOT       - R
print("RMSE: {}".format(r))
