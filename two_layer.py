
# coding: utf-8

# In[11]:


from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import tensorflow as tf


# In[12]:


mnist = input_data.read_data_sets(".", one_hot=True, reshape=False)
mnist


# In[13]:


learning_rate = 0.001
training_epochs =500
batch_size = 50  
display_step = 1

n_input = 784  
n_classes = 10  

n_hidden_layer1 = 256 
n_hidden_layer2 = 256


# In[14]:


weights = {
    'hidden_layer_1': tf.Variable(tf.random_normal([n_input, n_hidden_layer1])),
    'hidden_layer_2': tf.Variable(tf.random_normal([n_input, n_hidden_layer2])),
    'out': tf.Variable(tf.random_normal([n_hidden_layer2, n_classes]))
}
biases = {
    'hidden_layer1': tf.Variable(tf.random_normal([n_hidden_layer1])),
    'hidden_layer2': tf.Variable(tf.random_normal([n_hidden_layer2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}


# In[15]:


x = tf.placeholder("float", [None, 28, 28, 1])
y = tf.placeholder("float", [None, n_classes])

x_flat = tf.reshape(x, [-1, n_input])


# In[16]:


# Hidden layer 1 with RELU activation
layer_1 = tf.add(tf.matmul(x_flat, weights['hidden_layer_1']), biases['hidden_layer1'])
layer_1 = tf.nn.relu(layer_1)

# Hidden layer 2 with RELU activation
layer_2 = tf.add(tf.matmul(x_flat, weights['hidden_layer_2']), biases['hidden_layer2'])
layer_2 = tf.nn.relu(layer_2)

# Output layer with linear activation
logits = tf.matmul(layer_2, weights['out']) + biases['out']


# In[17]:


cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

init = tf.global_variables_initializer()


# In[18]:


with tf.Session() as sess:
    sess.run(init)

    for epoch in range(training_epochs):
        total_batch = int(mnist.train.num_examples/batch_size)
       
        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
           
            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
      
        if epoch % display_step == 0:
            c = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
            print("Epoch:", '%04d' % (epoch+1), "cost=","{:.9f}".format(c))
    print("Optimization Finished!")

    pred = tf.nn.softmax(logits)  # Apply softmax to logits
    
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))

    
    confusionmatrix = tf.contrib.metrics.confusion_matrix(tf.argmax(y,1),tf.argmax(pred,1))
    print('Confusion Matrix: \n\n', tf.Tensor.eval(confusionmatrix,feed_dict={x: mnist.test.images,                                                                              y: mnist.test.labels}, session=None))


# In[19]:


X = ([50, 125, 250, 500])
Y_50 = ([0.9125, 0.9297, 0.933, 0.9368])
Y_125 = ([0.9057, 0.9143, 0.9245, 0.9301])
Y_250 = ([0.8711, 0.8927, 0.9155, 0.9188])
Y_500 = ([0.8459, 0.8756, 0.8962, 0.9083])

plt.plot(X,Y_50,'b--', label = "BatchSize=50")
plt.plot(X,Y_125,'r--', label = "BatchSize=125")
plt.plot(X,Y_250,'g--', label = "BatchSize=250")
plt.plot(X,Y_500,'y--', label = "BatchSize=500")
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy as a function of Epoch_Two Layer_NN')
plt.legend(loc = "best")
plt.show()

