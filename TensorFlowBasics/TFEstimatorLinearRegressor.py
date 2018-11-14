import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from sklearn.model_selection import train_test_split as TTS

# TF
batch_size = 100
batches = 10000
lrate = 0.00001

# CONFIG
config = tf.ConfigProto(
	log_device_placement=True
)

# ONE MILLION POINTS
data_size = 1000000
x_data = np.linspace(0.0,10.0,data_size).astype('float32')

# NOISE
noise = np.random.randn(len(x_data))

# THE SLOPE AND INTERCEPT WE'RE TRYING TO ARRIVE AT
slope = 0.5
intercept = 5

# THE TARGET DATA (WITH NOISE)
y_true = (slope * x_data) + 5 + noise

# CREATE DATAFRAME
x_df = pd.DataFrame(data=x_data,columns=['X'])
y_df = pd.DataFrame(data=y_true,columns=['Y'])

my_data = pd.concat([x_df,y_df],axis=1)

# TRAIN TEST SPLIT
x_train,x_test,y_train,y_test = TTS(x_data,y_true,test_size=0.3)

t_x_train = tf.constant(x_train)
t_x_test = tf.constant(x_test)
t_y_train = tf.constant(y_train)
t_y_test = tf.constant(y_test)

ds_x_train = tf.data.Dataset.from_tensors(t_x_train)
df_x_test = tf.data.Dataset.from_tensors(t_y_train)
ds_y_train = tf.data.Dataset.from_tensors(t_y_train)
ds_y_test = tf.data.Dataset.from_tensors(t_y_test)

# FEATURE COLUMNS
feat_cols = [tf.feature_column.numeric_column('x',shape=[1])]

# THE ESTIMATOR - TENSORFLOW ABSTRACTION WHICH HANDLES THINGS FOR US
estimator = tf.estimator.LinearRegressor(feature_columns=feat_cols)

# INPUT FUNCTION
input_func = tf.estimator.inputs.numpy_input_fn({'x':x_train},y_train,batch_size=batch_size,num_epochs=None,shuffle=True)

train_input_func = tf.estimator.inputs.numpy_input_fn({'x':x_train},y_train,batch_size=batch_size,num_epochs=1000,shuffle=False)

test_input_func = tf.estimator.inputs.numpy_input_fn({'x':x_test},y_test,batch_size=batch_size,num_epochs=1000,shuffle=False)

# DO THE TRAINING
estimator.train(input_fn=input_func,steps = batches)

# EVALUATE THE TRAINING
train_metrics = estimator.evaluate(input_fn=train_input_func,steps=batches)

# COMPARE TO TEST DATA
test_metrics = estimator.evaluate(input_fn=test_input_func,steps=batches)

print("TRAINING DATA METRICS",train_metrics)
print("TEST DATA METRICS",test_metrics)

# PREDICTION
brand_new_data = np.linspace(0,10,10)
input_fn_predict = tf.estimator.inputs.numpy_input_fn({'x':brand_new_data},shuffle=False)

pred = list(estimator.predict(input_fn=input_fn_predict))

preds = [p['predictions'] for p in pred]

my_data.sample(250).plot(kind='scatter',x='X',y='Y')
plt.plot(brand_new_data,preds,'r')
plt.show()



