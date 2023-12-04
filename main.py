from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn import preprocessing
import numpy as np
import h5py
import matplotlib.pyplot as plt
from math import exp, log


def calculate_cost_LogReg(y, y_hat):
    """
    Calculates the cost of the OUTPUT OF JUST ONE pattern from the logistic
    regression classifier (i.e. the result of applying the h function) and
    its real class.
    
    Parameters
        ----------
        y: float
            Real class.
        y_hat: float
            Output of the h function (i.e. the hypothesis of the logistic
             regression classifier.
         ----------
    
    Returns
        -------
        cost_i: float
            Value of the cost of the estimated output y_hat.
        -------
    """
    cost_i=y*log(y_hat)+(1-y)*log(1-y_hat)
    return cost_i


def fun_sigmoid(theta, x):
    """
    This function calculates the sigmoid function g(z), where z is a linear
    combination of the parameters theta and the feature vector X's components
    
    Parameters
        ----------
        theta: numpy vector
            Parameters of the h function of the logistic regression classifier.
        x: numpy vector
            Vector containing the data of one pattern.
         ----------
    
    Returns
        -------
        g: float
            Result of applying the sigmoid function using the linear
            combination of theta and X.
        -------
    """
    
    return 1/(1+np.exp(-np.dot(theta,np.transpose(x))))


def train_logistic_regression(X_train, Y_train, alpha):
    """
    This function implements the training of a logistic regression classifier
    using the training data (X_train) and its classes (Y_train).

    Parameters
        ----------
        X_train: Numpy array
            Matrix with dimensions (m x n) with the training data, where m is
            the number of training patterns (i.e. elements) and n is the number
            of features (i.e. the length of the feature vector which
            characterizes the object).
        Y_train: Numpy vector
            Vector that contains the classes of the training patterns. Its
            length is m.

    Returns
        -------
        theta: numpy vector
            Vector with length n (i.e, the same length as the number of
            features on each pattern). It contains the parameters theta of the
            hypothesis function obtained after the training.

    """
    verbose = True
    max_iter = 500 
    # Number of training patterns.
    m = np.shape(X_train)[0]

    # Allocate space for the outputs of the hypothesis function for each
    # training pattern
    h_train = np.zeros(shape=(1, m))

    # Allocate spaces for the values of the cost function on each iteration
    J = np.zeros(shape=(1, 1 + max_iter))

    # Initialize the vector to store the parameters of the hypothesis function
    theta = np.zeros(shape=(1, 1 + np.shape(X_train)[1]))

    total_cost = 0
    for i in range(m):

        # Add a 1 (i.e., the value for x0) at the beginning of each pattern
        x_i = np.insert(np.array([X_train[i]]), 0, 1, axis=1)

        # Expected output (i.e. result of the sigmoid function) for i-th
        # pattern
        h_train[0,i]=fun_sigmoid(theta, x_i)
        # Calculate the cost for the i-the pattern and add it to the cost of
        # the last patterns
        total_cost = total_cost + calculate_cost_LogReg(Y_train[i], h_train[0,i])

    # b. Calculate the total cost
    total_cost=-1/m*total_cost
    J[0,0]=total_cost

    # GRADIENT DESCENT ALGORITHM TO UPDATE THE THETAS
    for num_iter in range(max_iter):

        gradient=np.zeros(shape=(m,1+np.shape(X_train)[1]))
        for i in range(m):
            # Add a 1 (i.e., the value for x0) at the beginning of each pattern
            x_i = np.insert(np.array([X_train[i]]), 0, 1, axis=1)

            # Expected output (i.e. result of the sigmoid function) for i-th
            # pattern
            h_train[0,i] = fun_sigmoid(theta, x_i)

            # Store h_i for future use
            gradient[i,:]=(h_train[0,i]-Y_train[i])*x_i
        theta=theta-alpha*(1/m)*sum(gradient)

        # Calculate the cost on this iteration and store it on vector J
        total_cost = 0
        for i in range(m):

            # Add a 1 (i.e., the value for x0) at the beginning of each pattern
            x_i = np.insert(np.array([X_train[i]]), 0, 1, axis=1)

            # Expected output (i.e. result of the sigmoid function) for i-th
            # pattern
            h_train[0,i]=fun_sigmoid(theta, x_i)
            # Calculate the cost for the i-the pattern and add it to the cost of
            # the last patterns
            total_cost = total_cost + calculate_cost_LogReg(Y_train[i], h_train[0,i])
        
        

        J[0,num_iter+1] = -1/m*total_cost

    if verbose:
        plt.plot(J[0], color='red')
        plt.xlabel('Number of iterations')
        plt.ylabel('Cost J')
        plt.title('learning rate = {}'.format(alpha))
        plt.show()

    return theta


def classify_logistic_regression(X_test, theta):
    """
    This function returns the probability for each pattern of the test set to
    belong to the positive class using the logistic regression classifier.

    Parameters
        ----------
        X_test: Numpy array
            Matrix with dimension (m_t x n) with the test data, where m_t
            is the number of test patterns and n is the number of features
            (i.e. the length of the feature vector that define each element).
        theta: numpy vector
            Parameters of the h function of the logistic regression classifier.

    Returns
        -------
        y_hat: numpy vector
            Vector of length m_t with the estimations made for each test
            element by means of the logistic regression classifier. These
            estimations corredspond to the probabilities that these elements
            belong to the positive class.
    """

    num_elem_test = np.shape(X_test)[0]
    y_hat = np.zeros(shape=(1, num_elem_test))

    for i in range(num_elem_test):
        # Add a 1 (value for x0) at the beginning of each pattern
        x_test_i = np.insert(np.array([X_test[i]]), 0, 1, axis=1)

        y_hat[0, i] = fun_sigmoid(theta, x_test_i)
        

    return y_hat

dir_output = "Output"
features_path = dir_output + "/features_geometric.h5"
labels_path = dir_output + "/labels_high-low.h5"
test_size = 0.3

# import features and labels
h5f_data = h5py.File(features_path, 'r')
h5f_label = h5py.File(labels_path, 'r')

features_string = h5f_data['dataset_inserts_geometric']
labels_string = h5f_label['dataset_inserts_geometric']

X = np.array(features_string)
Y = np.array(labels_string)

h5f_data.close()
h5f_label.close()


# SPLIT DATA INTO TRAINING AND TEST SETS
(X_train, X_test, Y_train, Y_test) = train_test_split(X, Y,
                                                      test_size=test_size,
                                                      random_state=1234)

# STANDARDIZE DATA
scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# print("Mean of the training set: {}".format(X_train.mean(axis=0)))
# print("Std of the training set: {}".format(X_train.std(axis=0)))
# print("Mean of the test set: {}".format(X_test.mean(axis=0)))
# print("Std of the test set: {}".format(X_test.std(axis=0)))

alpha = 1
theta = train_logistic_regression(X_train, Y_train, alpha)
Y_test_hat = classify_logistic_regression(X_test, theta)
Y_test_asig = Y_test_hat >= 0.5

# Show confusion matrix
Y_test = np.array([Y_test.astype(bool)])
confm = confusion_matrix(Y_test.T, Y_test_asig.T)
print(confm)
disp = ConfusionMatrixDisplay(confusion_matrix=confm)
disp.plot()

tn, fp, fn, tp = confm.ravel()
accuracy= (tp+tn)/(tp+tn+fp+fn)
precision=tp/(tp+fp)
recall=tp/(tp+fn)
f_score=2*(recall*precision)/(recall+precision)



print("***************")
print("The accuracy of the Logistic Regression classifier is {:.4f}".
      format(accuracy))
print("***************")


print("")
print("***************")
print("The precision of the Logistic Regression classifier is {:.4f}".
      format(precision))
print("***************")


print("")
print("***************")
print("The recall of the Logistic Regression classifier is {:.4f}".
      format(recall))
print("***************")


print("")
print("***************")
print("The F1-score of the Logistic Regression classifier is {:.4f}".
      format(f_score))
print("***************")

print(classification_report(Y_test.T, Y_test_asig.T))

tpr_list = [] 
fpr_list=[]
step=0.001
thresholds = list(np.arange(0.0, 1.0, step)) 
for threshold in thresholds:
    prediction= Y_test_hat >= threshold
    cfm= confusion_matrix(Y_test.T, prediction.T)
    tn, fp, fn, tp =cfm.ravel()
    tpr_list = [tp/ (tp + fn)] + tpr_list
    fpr_list = [fp / (fp + tn)] + fpr_list
plt.figure()
plt.plot(fpr_list, tpr_list)
plt.plot([0, 1], [0, 1], linestyle= 'dashed')
plt.xlabel("False Positive Rate") 
plt.ylabel("True Positive Rate") 
plt.title("ROC Curve")
plt.grid()
# AUC
auc = 0
for i in range(len(tpr_list)):
    if i < len(tpr_list) - 1:
        width = fpr_list[i+1] - fpr_list[i]
        triangle = width* (tpr_list[i+1] - tpr_list[i]) / 2
        rectangle = width*tpr_list[i]
        auc += triangle + rectangle
        
print("L'AUC est : %.4f" % auc)

from sklearn.tree import DecisionTreeClassifier

dec_tree = DecisionTreeClassifier(criterion = 'entropy', max_depth = 3)
dec_tree.fit(X_train, Y_train)
pred_tree = dec_tree.predict(X_test)

from sklearn import metrics

print("Accuracy (Decision Tree): ", metrics.accuracy_score(Y_test[0], pred_tree))
print("f1_score (Decision Tree): ", metrics.f1_score(Y_test[0], pred_tree))

from sklearn import svm

clf = svm.SVC(kernel='rbf')
clf.fit(X_train, Y_train) 
pred_svm = clf.predict(X_test)

print("Accuracy (SVM): ", metrics.accuracy_score(Y_test[0], pred_svm))
print("f1_score (SVM): ", metrics.f1_score(Y_test[0], pred_svm))

from sklearn.neighbors import KNeighborsClassifier
k = 4
neigh = KNeighborsClassifier(n_neighbors = k).fit(X_train, Y_train)
pred_knn = neigh.predict(X_test)

print("Accuracy (KNN): ", metrics.accuracy_score(Y_test[0], pred_knn))
print("f1_score (KNN): ", metrics.f1_score(Y_test[0], pred_knn))

from keras.models import Sequential
from keras.utils import to_categorical
from keras.layers import Dense

#Target
y_train = to_categorical(Y_train)
y_test = to_categorical(Y_test[0])

#Creating the NN
model = Sequential()
n_cols = X.shape[1]

model.add(Dense(5, activation = "relu", input_shape = (n_cols,)))
model.add(Dense(5, activation = "relu"))
model.add(Dense(5, activation = "relu"))
model.add(Dense(5, activation = "relu"))
model.add(Dense(2, activation = "softmax"))

# Compile the model
model.compile(optimizer = "adam", loss = "categorical_crossentropy", 
              metrics = ["accuracy"])

# build the model
model.fit(X_train, y_train, epochs = 20)

#Predicting
pred_train= model.predict(X_train)

pred_test= model.predict(X_test)
scores2 = model.evaluate(X_test, y_test, verbose = 0)
print('Accuracy on test data: {}% \n Error on test data: {}'.format(scores2[1], 1 - scores2[1]))    
