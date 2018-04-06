
# coding: utf-8

# In[31]:


import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt


# In[19]:


mnist = input_data.read_data_sets(".", one_hot=True)


# In[20]:


# learning_rate = 0.0001
# batch_size = 100
# display_step = 0.5
# a = np.zeros(shape=(10,10))

n_hidden_1 = 256 
n_input = 784 
n_classes = 10 


# In[21]:


X = tf.placeholder("float", [None, n_input])
Y = tf.placeholder("float", [None, n_classes])


# In[22]:


weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'out': tf.Variable(tf.random_normal([n_hidden_1, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}


# In[23]:


def RBF(x, C):
    return -tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(tf.expand_dims(x,2),
                                                   tf.expand_dims(C,0))),1))


# In[25]:


def multilayer_perceptron(x):
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
 
    layer_1 = tf.nn.relu(layer_1)
    return layer_1


input_X = multilayer_perceptron(X)

logits = RBF(input_X,weights['out'])+biases['out']


cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(logits,1), tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# In[29]:


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(8000):
        batch = mnist.train.next_batch(100)
        if i%100 == 0:
            train_accuracy = accuracy.eval(feed_dict={
                X:batch[0], Y: batch[1]})
            print("\repoch %d, training accuracy %g"%(i, train_accuracy), end="" if i%10 else "\n")
        train_step.run(feed_dict={X: batch[0], Y: batch[1]})
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Accuracy:", accuracy.eval({X: mnist.test.images, Y: mnist.test.labels}))
    
    pred = tf.nn.softmax(logits)  # Applying softmax to logits
         
    confusionmatrix = tf.contrib.metrics.confusion_matrix(tf.argmax(Y,1),tf.argmax(pred,1))
    print('Confusion Matrix: \n\n', tf.Tensor.eval(confusionmatrix,feed_dict={X: mnist.test.images, Y: mnist.test.labels}, session=None))


# In[33]:


X = ([0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 7900])
Y = ([0.04, 0.64, 0.78, 0.88, 0.89, 0.89, 0.87, 0.91, 0.9209])
plt.plot(X,Y,'b--')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy as a function of Epoch in RBF')
plt.legend(loc = "best")
plt.show()

