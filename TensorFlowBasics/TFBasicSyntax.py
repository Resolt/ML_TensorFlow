import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns

import tensorflow as tf

config = tf.ConfigProto(
	log_device_placement=True
)

with tf.Session(config=config) as sess:
	a = tf.random_normal((10000,10000))
	b = tf.random_normal((10000,10000))
	mm = tf.matmul(a,b)
	result = sess.run(mm)

print(result)


