import pandas as pd
import tensorflow as tf

dir = "TensorFlow/TensorFlowBasics/"
diabetes = pd.read_csv(dir+'pima-indians-diabetes.csv')

print(diabetes.head())

