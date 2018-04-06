# MNSIT-Classifiers
The primary goal of this project is to implement neural networks to classify handwritten digits. The handwritten digits are obtained from the open source MNIST database of handwritten digits. The MNIST database is a set of handwritten digits in which each digit is affixed in a 28x28 pixels bounding box. The images are black and white and every pixel represent a specific intensity of grayscale levels. The MNIST database consists of 60,000 training images and 10,000 testing images.
The project attempts to provide a neural networks based classification model that provides the lowest error rate. The classifiers implemented are:
•	Linear classifier
•	K-Nearest neighbor classifier
•	Radial Basis Function neural network
•	One-hidden layer fully connected multilayer neural network
•	Two-hidden layer fully connected multilayer neural network
•	Random Forest Classifier
•	Extra Trees Classifier
•	Multilayer Perceptron 

## Methodology for Classifiers  
1.The first step in any machine learning technique is to split the data for testing and training. Here the MNIST database is already been split into 60000 training images with corresponding labels and 10000 testing images with corresponding labels.

2.In the formulated code, the training data is further split into local internal training and validation data. The trained model is then tested on the global test set. 

2.FOR KNN, the training data is now fitted into KNN classifier by importing KNN classifier function from Scikit-Learn library. The value of K used is 10 here which is convention for data sets in the given range. The value of n_jobs is set to -1 indicating the number of parallel jobs to be performed is set to CPU cores.

3.The dataset is shuffled using the random function in the numpy library to accurately verify the validity of the results.

4.The prediction methodology for each classifier varies. 

5.The model is trained using the internal training dataset ( X_tr, y_tr). Then the results are verified using validation dataset ( X_val, y_val).

6.Finally the test data is fed into the model and the cross validation accuracy score is observed.
 
7.The performance metrics is inferred using the confusion matrix, implemented using the ‘confusion_matrix’ function. The confusion matrix is plotted in histogram matrix to display the accuracy of the model.

8.The performance metrics were obtained for the validation data. The performance metric used in this model is cross validation accuracy.

9.Finally, the metric is obtained for the test data.


|  Classifier        |    Accuracy using K-fold (k=3) .    | Error Rate  |
| ------------------ |:-----------------------------------:| -----------:|
| Linear             |         87.1%	                     |   12.9%     |
| K-Nearest Neighbor |          96.3%                      |    3.7%     |
| RBF Classifier     |         94%                         |      6%     |
| One-Hidden layer	 |         93%                         |      7%     |
| Two-Hidden Layer   |         93.6%	                     |    6.4%     |
| Random Forest      |         95%                         |      5%     |
| Extra Trees        |            94.3%                    |    5.7%     |
| MLP	               |         97.1%                       |    2.9%     |


The comparative analysis shows the performance metrics of each classifier. The linear classifier performs relatively poor due to the massive datasets and its inseparability. While, the MLP provides the highest accuracy of above 97.1%. Thus the performance of classification is dependent on the accurate choice of the model to be used. This is determined by the input data type. A simple dataset with relatively simpler separable data can be classified using linear classifier whereas a continuously varying data with high variance requires a convolutional neural network to obtain high accuracy.



