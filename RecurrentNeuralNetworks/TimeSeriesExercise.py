import numpy as np
import pandas as pd 
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler as SS

# READ DATA
dfile = '/home/resolt/Workspaces/MachineLearning/TensorFlow/RecurrentNeuralNetworks/monthly-milk-production.csv'
milk = pd.read_csv(dfile, index_col='Month')

# MAKE INDEX TIME SERIES
milk.index = pd.to_datetime(milk.index)

# PLOT OUT THE TIMESERIES DATA
# plt.plot(milk.index,milk['Milk Production'])
# plt.show()

# TRAIN TEST SPLIT
# X = milk['Milk Production'].head(milk.shape[1]-13)
# y = milk['Milk Production'].tail(12)
X = milk.head(milk.shape[0]-13)
y = milk.tail(12)

# SCALING
scaler = SS()
scaler.fit(X)
X = scaler.transform(X)
y = scaler.transform(y)

# FEEDER
def next_batch(training_data, batch_size, steps):
	"""
	INPUT: Data, Batch Size, Time Steps per batch
	OUTPUT: A tuple of y time series results. y[:,:-1] and y[:,1:]
	"""

	# STEP 1: Use np.random.randint to set a random starting point index for the batch.
	# Remember that each batch needs have the same number of steps in it.
	# This means you should limit the starting point to len(data)-steps
	rand_start = np.random.randint(0,X.shape[0]-steps-1) # THIS COULD NEED A MINUS ONE IN THE END

	# STEP 2: Now that you have a starting index you'll need to index the data from
	# the random start to random start + steps + 1. Then reshape this data to be (1,steps+1)


	# STEP 3: Return the batches. You'll have two batches to return y[:,:-1] and y[:,1:]
	# You'll need to reshape these into tensors for the RNN to .reshape(-1,steps,1)
