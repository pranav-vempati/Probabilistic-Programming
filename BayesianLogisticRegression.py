import edward as ed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from edward.models import Bernoulli, Normal, Empirical
import csv


csv_dataframe = pd.read_from_csv('data1.csv')

msg_id_array = np.array(csv_dataframe['messageID'].tolist())

labels = np.array(csv_dataframe['labels'].tolist())

input_dimensionality = msg_id.shape[0]

msg_id_array = msg_id_array.reshape((input_dimensionality,1))

output_size = 2

feature_dimensionality = 1

X_train = tf.placeholder(tf.float32, shape = (input_dimensionality, feature_dimensionality))

W = Normal(loc = tf.zeros(feature_dimensionality), scale = 2.0 * tf.ones(feature_dimensionality))

b = Normal(loc = tf.zeros([]), scale = 2.0 * tf.ones([]))

y = Bernoulli(logits = ed.dot(X_train,w) + b) 

qw_loc = tf.get_variable("qw_loc", [feature_dimensionality])

qw_scale = tf.nn.softplus(tf.get_variable("qw_scale", [feature_dimensionality]))

qb_loc = tf.get_variable("qb_loc", []) + 10.0 # Add a small bias constant

qb_scale = tf.nn.softplus(tf.get_variable("qb_scale", []))

qw = Normal(loc = qw_loc, scale = qw_scale)

qb = Normal(loc = qb_loc, scale = qb_scale)

inference = ed.KLqp({W:qw, b:qb}, data = {X_train:msg_id_array, y: labels}) # Use Variational Inference to leverage its fast convergence time. Hamiltonian Monte Carlo's burn in period is prohibitively long

inference.initialize(n_print = 10, n_iter = 500)

tf.global_variables_initializer().run()

for step in range(inference.n_iter):
	info_dict = inference.update()
	inference.print_progress(info_dict)

	
























