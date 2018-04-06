
# coding: utf-8

# In[26]:


from sklearn.datasets import fetch_mldata
from sklearn import model_selection

from sklearn.model_selection import cross_val_score, cross_val_predict, train_test_split
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
#from mnist import MNIST
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, precision_recall_curve
from sklearn.svm import LinearSVC
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.preprocessing import StandardScaler, Normalizer


# In[27]:


mnist = fetch_mldata('MNIST original')
X, y = mnist["data"], mnist["target"]

X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

shuffle_index = np.random.permutation(60000)
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]


# In[28]:


random_forest_clf = RandomForestClassifier(n_estimators=200, n_jobs=10,)
random_forest_clf.fit(X_train, y_train)


# In[29]:


cross_val_score(random_forest_clf, X_train, y_train, cv=3, scoring="accuracy")


# In[42]:


scaler = Normalizer()
X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))
cross_val_score(random_forest_clf, X_train_scaled, y_train, cv=3, scoring="accuracy")


# In[43]:


y_train_pred = cross_val_predict(random_forest_clf, X_train_scaled, y_train, cv=3)


# In[31]:


confusion_matrix = confusion_matrix(y_train, y_train_pred)
confusion_matrix


# In[32]:


def plot_confusion_matrix(matrix):
    """If you prefer color and a colorbar"""
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111)
    cax = ax.matshow(matrix)
    fig.colorbar(cax)

plt.matshow(confusion_matrix, cmap=plt.cm.gray)
plt.show()

row_sums = confusion_matrix.sum(axis=1, keepdims=True)
norm_conf_mx = confusion_matrix / row_sums
np.fill_diagonal(norm_conf_mx, 0)
plt.matshow(norm_conf_mx, cmap=plt.cm.gray)
plt.show()


# In[33]:


y_rf_pred = random_forest_clf.predict(X_test)

accuracy_score(y_test, y_rf_pred)


# In[34]:


print (precision_score(y_train, y_train_pred, average = None))


# In[35]:


print (precision_score(y_train, y_train_pred, average = 'weighted'))


# In[36]:


print(recall_score(y_train, y_train_pred, average = None))


# In[37]:


print(recall_score(y_train, y_train_pred, average = 'weighted'))


# In[38]:


print(f1_score(y_train, y_train_pred, average = None))


# In[39]:


print(f1_score(y_train, y_train_pred, average = 'weighted'))

