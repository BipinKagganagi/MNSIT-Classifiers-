
# coding: utf-8

# In[12]:


from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import tensorflow as tf


# In[13]:


mnist = input_data.read_data_sets(".", one_hot=True, reshape=False)
mnist


# In[14]:


# Parameters
learning_rate = 0.001
training_epochs = 500
batch_size = 50 
display_step = 1

n_input = 784  
n_classes = 10 

n_hidden_layer = 256 


# In[15]:


weights = {
    'hidden_layer': tf.Variable(tf.random_normal([n_input, n_hidden_layer])),
    'out': tf.Variable(tf.random_normal([n_hidden_layer, n_classes]))
}
biases = {
    'hidden_layer': tf.Variable(tf.random_normal([n_hidden_layer])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}


# In[16]:


x = tf.placeholder("float", [None, 28, 28, 1])
y = tf.placeholder("float", [None, n_classes])

x_flat = tf.reshape(x, [-1, n_input])


# In[17]:


# Hidden layer with RELU activation
layer_1 = tf.add(tf.matmul(x_flat, weights['hidden_layer']), biases['hidden_layer'])
layer_1 = tf.nn.relu(layer_1)

# Output layer with linear activation
logits = tf.matmul(layer_1, weights['out']) + biases['out']


# In[18]:


cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)


# In[19]:


# Initializing the variables
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    # Training cycle
    for epoch in range(training_epochs):
        total_batch = int(mnist.train.num_examples/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            # Run optimization op (backprop) and cost op (to get loss value)
            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
        # Display logs per epoch step
        if epoch % display_step == 0:
            c = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
            print("Epoch:", '%04d' % (epoch+1), "cost=","{:.9f}".format(c))
    print("Optimization Finished!")
    
    
    pred = tf.nn.softmax(logits)  # Apply softmax to logits
    
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
  
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
  
    
    confusionmatrix = tf.contrib.metrics.confusion_matrix(tf.argmax(y,1),tf.argmax(pred,1))
    print('Confusion Matrix: \n\n', tf.Tensor.eval(confusionmatrix,feed_dict={x: mnist.test.images,                                                                              y: mnist.test.labels}, session=None))


# In[20]:


X = ([100, 250, 500, 700])
Y_50 = ([0.9158, 0.9255, 0.9357, 0.9347])
Y_128 = ([0.8941, 0.9121, 0.9207, 0.9304])
Y_256 = ([0.8720, 0.8937, 0.9136, 0.9189])
Y_500 = ([0.8469, 0.8777, 0.8916, 0.9055])
Y_700 = ([0.8271, 0.8664, 0.8896, 0.8954])

plt.plot(X,Y_50,'b--', label = "BatchSize=50")
plt.plot(X,Y_128,'r--', label = "BatchSize=128")
plt.plot(X,Y_256,'g--', label = "BatchSize=256")
plt.plot(X,Y_500,'k--', label = "BatchSize=500")
plt.plot(X,Y_700,'y--', label = "BatchSize=700")
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy as a function of Epoch_One Layer_NN')
plt.legend(loc = "best")
plt.show()

