from numpy.linalg import pinv, norm
import numpy as np
from sklearn.metrics import confusion_matrix

class Classifier():
    def __init__(self, kernel, sigma = None):
        self.kernel = kernel
        self.sigma = sigma
            
    def _kernel(self, X1, X2):
        """
        Compute the kernel between X1 and X2
        
        Input
        ------
        X1: matrix of shape(n1_samples, n_features)
        
        X2: matrix of shape(n2_samples, n_features)
        
        
        Output
        -------
        Kernel matrix of shape(n1_samples, n2_samples)
        
        """
              
        if self.kernel == 'rbf':
            """
            K = exp(-1/2.sigma^2 ||X1 - X2||^2)
            
            """
            
            K = np.zeros((X1.shape[0], X2.shape[0]))
            for i, x1 in enumerate(X1):
                for j, x2 in enumerate(X2):
                    K[i, j] = np.exp(-(1/(2*self.sigma**2))*norm(x1 - x2)**2)
            return K
        
        elif self.kernel == 'linear':
            """
            K = X*X.T
            
            """
            
            K = np.dot(X1, X2.T) 
            return K
        
    def fit(self, X, y_train):
        """
        Calculate the weights alpha
        
        alpha = inv(K). y
        
        K = Kernel matrix
        
        If the eigenvalues of the kernel matrix is close, it would cause instability in the model.
        Hence an identity matrix with small coefficient could help us to gain better stability.
        So, alpha can be calculated as;
        
        alpha = inv(K + gamma.I).y
        
        gamma = small coefficient
        I = Identity matrix
        
        
        Input
        ------
        X: matrix of shape(n_samples, n_features)
        
        y_train: vector of length(n_samples)
        
               
        """
        
        self.X_train  = X
        # K = self._kernel(self.X_train, self.X_train) + 1
        K = self._kernel(self.X_train, self.X_train)
        # I = pinv(K + 0.01*np.identity(len(y_train)))
        I = pinv(K)
        self.alpha = np.dot(I, y_train)
        
    def predict(self, X):
        """
        Predict the output for test set based on the weights calculated in method 'fit'
        
        Input
        -----
        X: test data(matrix of shape(n_samples, n_features))
        
        
        Output
        ------
        y_p: predicted output(array of length n_samples)
        
        """
        
        X_test = X
        K_ = self._kernel(self.X_train, X_test)
        y_pred = np.dot(self.alpha, K_)   
        # y_pred gives continuous values instead of actual labels we are looking for.
        
        # converting the continuous values to labels
        self.y_p = [-1 if x <= 0 else 1 for x in y_pred]
        return self.y_p 
    
    def evaluate(self, y_test):        
        """
        Exaluate the performance of the model in terms of accuracy and precision
        
        Input
        ------
        y_test: actual label(an array of length n_samples)
        
        output
        ------
        Accuracy and precision of the classifier model
        
        """
        
        # Calculating confusion matrix
        cm = confusion_matrix(y_test, self.y_p)
        accuracy = (cm[0][0] + cm[1][1])/(sum(sum(cm)))
        precision = cm[0][0] / (cm[0][0] + cm[0][1])
        return accuracy, precision