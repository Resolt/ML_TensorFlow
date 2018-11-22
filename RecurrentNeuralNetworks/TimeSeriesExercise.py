import numpy as np
import pandas as pd 
import tensorflow as tf

# READ DATA
dfile = '/home/resolt/Workspaces/MachineLearning/TensorFlow/RecurrentNeuralNetworks/monthly-milk-production.csv'
df = pd.read_csv(dfile, index_col='Month')

# MAKE INDEX TIME SERIES
milk.index = pd.to_datetime(milk.index)


