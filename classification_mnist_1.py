import numpy as np
from sklearn.datasets import fetch_mldata
import matplotlib
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_predict

mnist = fetch_mldata('MNIST original')

x, y, z = mnist["data"],mnist["target"],mnist["DESCR"]
#print(x.shape, y.shape)

some_digit = x[36000]
#print(type(some_digit))
some_digit_image = some_digit.reshape(28,28)
plt.imshow(some_digit_image, cmap=matplotlib.cm.binary, interpolation='nearest')
#plt.show()

x_train, x_test, y_train, y_test = x[:60000], x[60000:], y[:60000], y[60000:]
shuffle_index = np.random.permutation(60000)
x_train, y_train = x_train[shuffle_index], y_train[shuffle_index]
y_train_5 = y_train == 5
y_test_5 = y_test == 5

"""############# BINARY  CLASSIFIER ##############"""
sgd_clf = SGDClassifier(random_state=2)
sgd_clf.fit(x_train, y_train_5)
print(sgd_clf.predict([some_digit]))
y_train_predict = cross_val_predict(sgd_clf, x_train, y_train_5, cv=3 )
print(y_train_predict)

''' ########## MULTICLASS CLASSIFIER ##########'''
''' 
normally this create one vs all strategy.
it means when considering this it create 10 binary classifier such that 5 0r not
6 or not like that.
from those given the higher scored class as predicted class
'''
sgd_clf.fit(x_train, y_train)
print(sgd_clf.predict([some_digit]))

'''##### ONE VS ONE CLASSIFIER #####'''

from sklearn.multiclass import OneVsOneClassifier

ovoClassifier = OneVsOneClassifier(SGDClassifier(random_state=4))
ovoClassifier.fit(x_train, y_train)

'''
in this problem when we select one vs one classifier there are 45 estimators
estimators are build predit whether 1 or 2 , 1 or 3, 1 or 4, like that 
'''
print(len(ovoClassifier.estimators_))

'''###### MULTILABEL CLASSIFIER ######'''

from sklearn.neighbors import KNeighborsClassifier

y_train_large = (y_train >= 7)
y_train_odd = (y_train % 2 == 1)
y_multilabel = np.c_[y_train_large, y_train_odd]
km_clf = KNeighborsClassifier()
#km_clf.fit(x_train, y_multilabel)
''' ###### MULTIOUTPUT CLASSIFICATION ######'''

noise_train = np.random.randint(0, 100, (len(x_train), 784))
noise_test = np.random.randint(0, 100, (len(x_test), 784))
x_train_mod = x_train + noise_train
x_test_mod = x_test + noise_test
y_train_mod = x_train
y_test_mod = x_test
print(x_train_mod.shape)
print(x_test_mod.shape)
print(x_test_mod[200].shape)

km_clf.fit(x_train_mod, y_train_mod)
#print(x_test_mod[some_digit].shape)
cleanDigit = km_clf.predict([x_test_mod[3600]])
#print(cleanDigit.shape)
image_resize = cleanDigit.reshape(28, 28)
plt.imshow(image_resize, cmap=matplotlib.cm.binary)
plt.show()