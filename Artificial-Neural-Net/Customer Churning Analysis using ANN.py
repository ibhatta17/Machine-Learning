# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Since we have some categorical variable in ur dataset, we need to map them to numerical values for ML processing
# Encoding the categorical variable
# Encoding the Independent Variable(For Geography and Gender)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
# In order to make sure there is no hierarchy for categorical variables
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
# This avoids dummy variable trap(for n category, we just need n-1 dummy variable)
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split # cross_validation is replaced by model_selection
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Importing Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initializing the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
# we will have 11 input nodes(for 11 independent variables)
# The best activation function could be rectifier function for the hidden layer and sigmoid function for the output layer
# We will also be able to find the rank of probability that customer could leave the bank
classifier.add(Dense(output_dim = 8,# number of nodes in the hidden layer being added. Usual practice is to take an 
# average of number of layers in i/p layer and o/p layers. Or performance tuning by K-fold cross validation 
init = 'uniform' , # initialize the weights to small number close to 0
activation = 'relu', # rectifier activation function for hidden layer
input_dim = 11 # number of nodes in th i/p layer( # of independent variable)
)) # all the NN parameters are defined here

# Adding additional hidden layer
classifier.add(Dense(output_dim = 8, init = 'uniform', activation = 'relu')) 
# input_dim is required only for first hidden layer as the NN model does not know how many nodes are at the input but after 
# first hidden layer, the model knows how many input are there in the following hidden layers

# Adding additional hidden layer
classifier.add(Dense(output_dim = 4, init = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(output_dim = 1, # since we are expecting binary output, we need 1 output node. 
init = 'uniform', 
activation = 'sigmoid' # sigmoid activation function for output layer
# for muti-value output categories, we need output_dim = nimber of categories and activation = 'soft_max'
# soft_max is similar to sigmoid function but applied to dependent varibale that has more than 2 categories
)) 

# Compiling the ANN
# Applying stochastic gradient descent in the entire NN
classifier.compile(
optimizer = 'adam', # algorithm to find optimal set of weights for NN
loss = 'binary_crossentropy', # loss function within the stochastic gradient descent algorithm (i.e. in 'adam' algorithm
# binary_crossentropy -> for binary o/p and categorical_crossentropy -> for categorical o/p
metrics = ['accuracy'] # accuracy criterion to evaluate the model
)

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, 
batch_size = 10, # whether to update the weights after each observation or after a batch of obobservation ckpropagation
nb_epoch = 100 # defines number of iterations
# for both batch_size and nb_epoch, no optimal value by default. Need to find the best value by experimentation or performance tuning
)

#Predicting the Test set results
y_pred = classifier.predict(X_test)

# since the y_pred here is the probabilities instead of binary value, we need to convert these probabilities into a binary value. 
# For this weed to set a threshold to distinguish between 1 and 0.
# for sensitive information we need higher threshold
# let's choose 50% as the threshold here
y_pred = (y_pred > 0.5) # this gives true/false

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
# cm

# Computing the Test Accuracy
test_accuracy = (cm[1,1] + cm[0,0]) / sum(sum(cm))
print(test_accuracy)
# Computing the Test Precision
test_precision = cm[1,1] / (cm[1,1] + cm[0,1])
print(test_precision)