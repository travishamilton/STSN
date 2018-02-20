import numpy as np
import tensorflow as tf
import os

import pandas as pd

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'				# clears Tensorflow CPU for my mac Unix terminal
os.system('cls' if os.name == 'nt' else 'clear')	# clears the terminal window screen (clc equiv. to MATLAB)

np.random.seed(7)		# seeding the random number generator to reproduce identical results
tf.set_random_seed(7)	# seed Tensorflow random numebr generator as well


#------------------------ Read in Data ------------------------#
userScatter = raw_input("Scatter-points: ")
userTime = raw_input("Time units: ")
fileName = 'data/scatter' + userScatter + '_T' + userTime + '_all_'
X = np.transpose( np.genfromtxt(fileName + 'in.csv', delimiter = ',') )
# X = np.ones((15,16))
Y = np.transpose( np.genfromtxt(fileName + 'out.csv', delimiter = ',') )
# Y = np.random.random((15,16))

layers = fileName[-7:-5]	# read in the number of layers from the file name
layers = int(layers)		# number of scatter/prop. layers to navigate through

print("This model contains:")
print("\t- " + str(layers) + " time units")
print("\t- " + str(int(fileName[-11:-9])) + " scatter weights\n")

sampN, featN = X.shape	# sampN: number of training samples, featN: features per sample 


#----------- Random Weight Generation For Material -----------#

wN = featN//2					# number of trainable transmission weights

# ---------------------- Propagation Layer ----------------------#
def propagate(x):
	sampN, featN = x.shape.as_list()	# get shape of the data, O(1) constant run-time

	# each iteration of this for-loop performs an O(n) slice along a column, therefore
	# total run-time is O(200n) = O(n)
	for i in range(featN):
		
		# if the first feature, transmit nothing
		if i is 0:
			out = tf.zeros(shape = [sampN,1], dtype = tf.float64)
			# to avoid concatenation, we have to continue to the next iteration
			continue
		
		# if the last feature, transmit nothing
		elif i is featN-1:
			nex = tf.zeros(shape = [sampN,1], dtype = tf.float64)
			
		# otherwise, if at odd index, transmit input from southwest (according to figure)
		elif i % 2:
			nex = x[:,i+1]
			nex = tf.reshape( nex,(nex.shape.as_list()[0],1) )	# must reshape because slicing interprets as 1-D column
			
		# else if at even index, transmit input from northwest (according to figure)
		else:
			nex = x[:,i-1]
		
		# must reshape because slicing interprets as 1-D column
		nex = tf.reshape( nex,(nex.shape.as_list()[0],1) )

		# every time we obtain the "next" column, append it (tried using tf.stack, too confusing)
		out = tf.concat([out, nex], axis = 1)
	
	return out


#------------------------ Scatter Layer ------------------------#
def scatter(x,w):
	
	sampN, featN = x.shape.as_list()	# get input dimensions
	
	# slicing in Tensorflow calls tf.slice automatically because of operator-overloading
	evenMat = x[:,::2]	# take the even-indexed values
	oddMat = x[:,1::2]	# ... .... odd ... ...
	
	'''
		In this function, w takes the shape of a (N x 1) row vector, i.e. wN = 3 (number of trainable weights)
					
				W = [1  1  1]

		Performing this "tile function repeats W for every data "sample" we have so we don't
		have to loop and keep reusing W. Instead we have a matrix of repeating rows, wTile,
		and do the element-wise multiplication directly all at once. Remember that each row 
		refers to each new data sample. If the number of training samples, sampN, is 5, we'll have:

					[1  1  1]
					[1  1  1]
			wTile =	[1  1  1]
					[1  1  1]
					[1  1  1]

		Doing this saves us the headache of using nested for loops and keeps the runtime
		complexity exactly the same unless Tensorflow's backend code has element-wise
		matrix multiplication optimized to run faster than O(n*m)
	'''

	wTile = tf.tile(w, [sampN,1])	# create matrix of repeating rows
	oneMinus = 1 - wTile 			# subtracting the repeating rows matrix by one is the same as tf.ones(.) - wTile
	
	''' !!!!!!!!!!!!!!!!!!!!!!!!!!!!
		CAUTION Do not get confused: 
			- 	tf.multiply(,) is actually element-wise multiplication (that's why I used it)
			- 	tf.matmul(,) is matrix multiplication
			- 	when you do A*B, I'm not sure what Tensorflow's '*' operator is using. In other words,
				we don't have access to see what operator overloading '*' does when A*B is being computed.
	'''

	# top outputs for each "cross" (sorry I couldn't think of better wording)
	top =  tf.multiply(oddMat,wTile) + tf.multiply(evenMat,oneMinus)
	
	# bottom ... ... ... ...
	bottom = tf.multiply(evenMat,wTile) + tf.multiply(oddMat,oneMinus)
	
	# iterate through each feature of the top and bottom matrices
	for i in range(featN//2):

		# if the first iteration is reached, instantiate the output matrix with the first column of top
		if i is 0:
			out = top[:,i]
			out = tf.reshape( out, (out.shape.as_list()[0],1) )	# must reshape because slicing interprets as 1-D column

		# otherwise just append the next element of top to the already instantiated output
		else:
			nex = top[:,i]
			nex = tf.reshape( nex,(nex.shape.as_list()[0],1) )	# must reshape because slicing interprets as 1-D column

			# every time we obtain the "next" column, append it (I tried using tf.stack (synonymous to np.stack) but it was too confusing)
			out = tf.concat([out, nex], axis = 1)

		# at the end of every iteration, append the i-th element of bottom
		nex = bottom[:,i]
		nex = tf.reshape( nex,(nex.shape.as_list()[0],1) )	# must reshape because slicing interprets as 1-D column
		out = tf.concat([out, nex], axis = 1)

	return out


#-------------------------- Transmision ------------------------#
def transmit(x, w, N):
	
	for i in range(N):
		if i is 0:
			out = x
		out = scatter(out,w)
		out = propagate(out)

	return out

#--------------------------- Variable Instantiation --------------------------#
X_tens = tf.placeholder(dtype = tf.float64, shape = [sampN,featN])
Y_tens = tf.placeholder(dtype = tf.float64, shape = [sampN,featN])
W_tens = tf.Variable(tf.ones(shape = [1,featN//2], dtype = tf.float64))


#--------------------------- Cost Function Definition --------------------------#
# compute least squares cost for each sample and then average out their costs
print("Building Cost Function (Least Squares) ... ... ...")
Yhat_tens = transmit(X_tens, W_tens, layers)
# least_squares = tf.reduce_sum( tf.reduce_sum((Yhat_tens - Y_tens)**2, axis = 1) ) / featN
least_squares = tf.norm((Yhat_tens - Y_tens)**2, ord=2)**2 / featN * 200
print("Done!\n")

#--------------------------- Define Optimizer --------------------------#
print("Building Optimizer ... ... ...")
train_op = tf.train.AdamOptimizer(0.08).minimize(least_squares)
print("Done!\n")



#--------------------------- Training --------------------------#
epochs = 2000
table = []

with tf.Session() as sess:
	sess.run( tf.global_variables_initializer() )
	
	# I did all this tf.convert_to_tensor(.) crap to print out information before training for my own sanity
	print("Tensor X:")
	print(X)
	print("")

	print("Tensor W: ") 
	print(W_tens.eval())
	print("")

	print("Tensor Y: ")
	print(Y)
	print("")

	print("--------- Starting Training ---------\n")
	for i in range(1, epochs+1):
		_, loss_value = sess.run([train_op, least_squares], feed_dict = {X_tens: X, Y_tens: Y})

		if i == 1 or i % 5 == 0:
			currStatus = [i, loss_value]
			for w in W_tens.eval()[0]:
				currStatus.append(w)
			table.append( currStatus )

			print("Epoch: " + str(table[-1][0]) + "\t\tLoss: " + str(table[-1][1]))
			print(W_tens.eval())


headers = ['Epoch', 'Loss']
for i in range(1,wN+1):
	headers.append('W'+str(i))

df = pd.DataFrame(table, columns = headers)
df.to_csv("results/Unmasked Losses_" + fileName[-11:-1] + ".csv")