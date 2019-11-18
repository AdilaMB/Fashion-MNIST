# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

trainig = pd.read_csv('E:/Maestrado/Machine Learning/Machine Learning I/DataSet/MNIST/fashion-mnist_train.csv',
                              sep=';', header=None)
test = pd.read_csv('E:/Maestrado/Machine Learning/Machine Learning I/DataSet/MNIST/fashion-mnist_test.csv',
                       sep=';', header=None)


    # Divido los datos del trining y test
X_trainig = trainig.iloc[1:, 1:].values
Y_trainig = trainig.iloc[1:, 0:1].values.reshape(-1, 1)

Y_trainig = Y_trainig.astype('int')
X_trainig = X_trainig.astype('int')

X_test = test.iloc[1:, 1:].values
Y_test = test.iloc[1:, 0:1].values.reshape(-1, 1)

X_test = X_test.astype('int')
Y_test = Y_test.astype('int')

## 30% de teste
x_train, x_eval, y_train, y_eval = train_test_split(X_trainig, Y_trainig, test_size=0.3, random_state=100)

    # Parameters
learning_rate = 0.001
training_epochs = 15
batch_size = 100
display_step = 1


    # Parametros de la red
n_hidden_1 = 512 # 1st layer number of neurons
n_hidden_2 = 256 # 2nd layer number of neurons
n_input = x_train.shape[0] # data input 42 000
n_classes = y_train.shape[1] # total classes (0-9 digits)

num_output = y_train.shape[1]
regularizer_rate = 0.1
dropout_prob = 0.6


    # Construyendo el grafo con sus tensores
X = tf.placeholder("float",[None, n_input]) #matriz de [0][42000]
#tf.squeeze(x, axis=[0])
tf.to_float(X)
Y = tf.placeholder("float", [None, n_classes])
#tf.to_float(Y)
keep_prob = tf.placeholder(tf.float32)

    # Inicializando los pesos y los factores de penalidad

#weights_h1 = tf.Variable(tf.random_normal([n_input, n_hidden_1], stddev=(1 / tf.sqrt(float(n_input))))) #[42000, 512]
weights_h1 = tf.Variable(tf.random_normal([n_input, n_hidden_1]))
weights_h2 = tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2], stddev=(1 / tf.sqrt(float(n_hidden_1)))))
weights_out = tf.Variable(tf.random_normal([n_hidden_2, n_classes], stddev=(1 / tf.sqrt(float(n_hidden_2)))))

biases_b1 = tf.Variable(tf.random_normal([n_hidden_1]))
biases_b2 = tf.Variable(tf.random_normal([n_hidden_2]))
biases_out = tf.Variable(tf.random_normal([n_classes]))
    # Creando el modelo Sumatoria de W*x+ b
    # Hidden fully connected layer with 256 neurons
layer_1 = tf.add(tf.matmul(X, weights_h1), biases_b1)
layer_1 = tf.nn.relu(layer_1)

layer_2 = tf.add(tf.matmul(layer_1, weights_h2), biases_b2)
layer_2 = tf.nn.relu(layer_2)

out_layer = tf.add(tf.matmul(layer_2, weights_out), biases_out)
out_layer = tf.nn.relu(out_layer)

   #La funcion de perdida
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
logits=out_layer, labels=Y))+ regularizer_rate*(tf.reduce_sum(tf.square(biases_b1)) + tf.reduce_sum(tf.square(biases_b2)))

    # Backpropagation (using Adam optimizer)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

        #return optimizer, loss_op

       # Initializing the variables
init = tf.global_variables_initializer()

    # Start session to compute Tensorflow graph
with tf.Session() as sess:
    # Run initialization
    sess.run(init)

        # Training cycle
    for epoch in range(training_epochs):

        avg_cost = 0.
        total_batch = int(trainig.shape[0]/batch_size)
        arr = np.arange(x_train.shape[0])
        np.random.shuffle(arr)
        iter_cost = 0.

        for i in range(0, x_train.shape[0], total_batch):
            batch_x = tf.data.Dataset.batch(total_batch, x_train)
            batch_y = tf.data.Dataset.batch(total_batch, y_train)

            # for (minibatch_X, minibatch_Y) in tf.data.Dataset.zip(batch_x, batch_y):
           # minibatch_X, minibatch_Y = np.asmatrix(minibatch_X), np.asmatrix(minibatch_Y)
            #minibatch_cost, acc = sess.run(optimizer, feed_dict={X: batch_x, Y: batch_y})
            #iter_cost += minibatch_cost * 1.0 / total_batch

    #testing_accuracy.append(accuracy_score(y_test.argmax(1),
                                          # s.run(predicted_y, {input_X: X_test, keep_prob: 1}).argmax(1)))

    # Loop over all batches ##grads_and_vars = opt.compute_gradients(loss, <list of variables>)
            sess.run(optimizer,{Y: y_train[arr[i:i + batch_size]], X: x_train[arr[i:i + batch_size]]
                              })

            #sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})

        training_accuracy.append(sess.run(accuracy, feed_dict={X: x_train, Y: y_train}))
        training_loss.append(sess.run(loss, {X: x_train, Y: y_train}))
           # Compute average loss
        testing_accuracy.append(accuracy_score(y_test.argmax(1),
                                               s.run(predicted_y, {input_X: X_test}).argmax(1)))
        # Display logs per epoch step
if epoch % display_step == 0:
                #print("Epoch:", '%04d' % (epoch+1), "cost={:.9f}".format(avg_cost))
    print("Optimization Finished!")


        # Test model
pred = tf.nn.softmax(logits)  # Apply softmax to logits
correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))

        # Calculate accuracy
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

print("Accuracy:", accuracy.eval({X: x_eval, Y: y_eval}))


