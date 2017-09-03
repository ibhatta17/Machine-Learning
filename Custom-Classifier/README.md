# Custom Build SVM Classifier From Scratch

The goal of this project is to build a classifier froam a scratch instead of using the one that comes with Scikit-learn. Then, use the classifier to classify Fibonacci and XOR data. 

![Fibonacci Data](/img/Fibonacci.png)

In here, SVM classifier is developed using linear and RBF kernel.

For a linear classifiier,
* y_i = w^TX_i + b *


### K-Fold Cross Validation
Any machine learning model involves a lot of random selection throughout the process. We may get a higher accuracy when we run a model. But, when we use the same model next time, we may get get much lower accuracy. So, instead of decising the performance of a model just from one iteration, we can run the model multiple times & calculate the error from each iteration and then calculate average error. This approach will give us a more accurate performance measure.

To acheive this, K-fold cross validation technique is used in this project. The value of K is chosen to be 5. Meaning, each classifier is trained and evaluate 5 times before measuring accuracy of the model.


### Grid Search Approach
There are many hyperparameters that needs to be selected before using them in a classifier. We normally choose these parameters randomly using our intuition. But we never know which selection gives us the most accurate model. Grid search method is a great way determine the best selection of hyperparameters.

Grid search is a method in which we create a grid of parameters. Then we use each possible combinations of parameters in the classifier  and train and evaluate the model. Finally, we select the parameters which produces least classification error.
