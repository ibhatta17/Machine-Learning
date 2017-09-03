from sklearn.model_selection  import train_test_split
from classifier import Classifier

class GridSearch():
    def __init__(self, param_grid, k = 5, measure = 'accuracy'):
        self.param_grid = param_grid
        self.k = k # number of folds for cross validations
        self.measure = measure        
        
    def _define_classifiers(self):
        """
        Define classifiers with all possible combinations of parameters in param_grid
        
        Output
        -------
        classifiers: a list of classifiers
        
        """
        classifiers = []
        for param in self.param_grid:
            kernel = param['kernel'][0]
            if kernel == 'linear':
                classifiers.append(Classifier(kernel = 'linear'))
            elif kernel == 'rbf':
                s = param['sigma']
                for i in range(len(s)):
                    classifiers.append(Classifier(kernel = 'rbf', sigma = s[i]))
        return classifiers        
        
    def fit(self, X, y):
        """
        Find the best combination of parameters that cause lowest classification error
        
        Input
        ------
        X: training data(matrix of n_samples and n_features)
        
        y: training label(vector of length n_samples)
        
        Output
        -------
        best_score: the best accuracy obtained from all possible combinations of hyperparameters provided in param_grid
        
        best_classifier: classifier that is associated which produces best_score
        
        """
        classifiers = self._define_classifiers()
        best_score = 0
        for cl in classifiers: # iterating over each comibation from the grid        
            score = []
            for _ in range(self.k): # iterating over K-folds for cross validation
                # Splitting the dataset into the Training set and Validation set
                X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.25)
                # using 75% of the data for training and 25% for validating the model
                cl.fit(X_train, y_train)
                y_pred = cl.predict(X_val)

                accur, prec  = cl.evaluate(y_val)
                if self.measure == 'accuracy':
                    score.append(accur)
                elif self.measure == 'precision':
                    score.append(prec)

            avg_score = sum(score)/len(score)

            if avg_score > best_score:
                best_score = avg_score
                best_classifier = cl
        return best_score, best_classifier       