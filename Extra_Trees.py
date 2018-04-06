
# coding: utf-8

# In[13]:


from sklearn.datasets import fetch_mldata
from sklearn import model_selection

from sklearn.model_selection import cross_val_score, cross_val_predict, train_test_split
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.svm import LinearSVC
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.preprocessing import StandardScaler


# In[14]:


mnist = fetch_mldata('MNIST original')
X, y = mnist["data"], mnist["target"]

X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

shuffle_index = np.random.permutation(60000)
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]


# In[19]:


some_digit = X_train[5000]
# some_digit_image = some_digit.reshape(28, 28)
# plt.imshow(some_digit_image, cmap = matplotlib.cm.binary,interpolation="nearest")
# plt.show()


# In[20]:


clf_et = ExtraTreesClassifier(n_estimators=200, n_jobs=10,)
clf_et.fit(X_train, y_train)

clf_et.predict([some_digit])


# In[21]:


cross_val_score(clf_et, X_train, y_train, cv=3, scoring="accuracy")


# In[22]:


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))


# In[38]:


cross_val_score(clf_et, X_train_scaled, y_train, cv=3, scoring="accuracy")


# In[39]:


y_train_pred = cross_val_predict(clf_et, X_train_scaled, y_train, cv=3)


# In[26]:


confusion_matrix = confusion_matrix(y_train, y_train_pred)
print(confusion_matrix)


# In[27]:


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


# In[28]:


y_et_pred = clf_et.predict(X_test)

accuracy_score(y_test, y_rf_pred)


# In[29]:


print (precision_score(y_train, y_train_pred, average = None))


# In[30]:


print (precision_score(y_train, y_train_pred, average = 'weighted'))


# In[31]:


print(recall_score(y_train, y_train_pred, average = None))


# In[32]:


print(recall_score(y_train, y_train_pred, average = 'weighted'))


# In[33]:


print(f1_score(y_train, y_train_pred, average = None))


# In[34]:


print(f1_score(y_train, y_train_pred, average = 'weighted'))

